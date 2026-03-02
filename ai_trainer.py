import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from collections import Counter
import re
import random
import json

# ------------------------------
# 1. Подготовка данных
# ------------------------------

# Предположим, у вас есть файл data.txt с парами instruction и answer, разделёнными табуляцией или специальным разделителем.
# Пример строки: "инструкция: напиши стихотворение\tответ: вот стихотворение..."
# Замените этот код на загрузку вашего файла.

def load_data(file_path):
    instructions = []
    answers = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                instructions.append(data['instruction'])
                answers.append(data['output'])
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Ошибка в строке: {line}\n{e}")
    return instructions, answers

# Здесь укажите путь к вашему файлу
instructions, answers = load_data('genshin_qa.jsonl')

# # Для примера создадим синтетические данные (замените на реальные)
# instructions = [
#     "напиши приветствие",
#     "сколько будет 2+2",
#     "расскажи шутку"
# ]
# answers = [
#     "привет, рад тебя видеть!",
#     "2+2 равно 4",
#     "почему программисты путают хэллоуин и рождество? потому что 31 oct == 25 dec"
# ]

# ------------------------------
# 2. Токенизация и построение словаря
# ------------------------------

# Простая токенизация по словам (можно заменить на более сложную)
def tokenize(text):
    # удаляем знаки препинания и приводим к нижнему регистру
    text = re.sub(r'[^\w\s]', '', text.lower())
    return text.split()

# Собираем все слова из инструкций и ответов
all_tokens = []
for inst, ans in zip(instructions, answers):
    all_tokens.extend(tokenize(inst))
    all_tokens.extend(tokenize(ans))

# Строим частотный словарь и оставляем только наиболее частые слова (например, топ-5000)
vocab_size = 5000
word_counts = Counter(all_tokens)
most_common = word_counts.most_common(vocab_size - 4)  # резервируем место для спецтокенов

# Создаём отображения слово -> индекс и индекс -> слово
word2idx = {'<PAD>': 0, '<SOS>': 1, '<EOS>': 2, '<UNK>': 3}
idx2word = {0: '<PAD>', 1: '<SOS>', 2: '<EOS>', 3: '<UNK>'}

for idx, (word, _) in enumerate(most_common, start=4):
    word2idx[word] = idx
    idx2word[idx] = word

vocab_size = len(word2idx)

# Функция для преобразования текста в последовательность индексов
def encode(text, max_len=None):
    tokens = tokenize(text)
    ids = [word2idx.get(token, word2idx['<UNK>']) for token in tokens]
    if max_len is not None:
        if len(ids) > max_len:
            ids = ids[:max_len]
        else:
            ids = ids + [word2idx['<PAD>']] * (max_len - len(ids))
    return ids

def decode(ids):
    return ' '.join([idx2word.get(idx, '<UNK>') for idx in ids if idx not in (0,1,2)])

# ------------------------------
# 3. Создание Dataset и DataLoader
# ------------------------------

class InstructionAnswerDataset(Dataset):
    def __init__(self, instructions, answers, max_len_in=20, max_len_out=20):
        self.instructions = instructions
        self.answers = answers
        self.max_len_in = max_len_in
        self.max_len_out = max_len_out

    def __len__(self):
        return len(self.instructions)

    def __getitem__(self, idx):
        inst = self.instructions[idx]
        ans = self.answers[idx]
        # кодируем инструкцию (вход энкодера)
        enc_in = encode(inst, self.max_len_in)
        # для декодера вход: <SOS> + ans, выход: ans + <EOS>
        ans_ids = encode(ans, self.max_len_out - 1)  # оставляем место для <EOS>
        dec_in = [word2idx['<SOS>']] + ans_ids
        dec_out = ans_ids + [word2idx['<EOS>']]
        # паддинг до одинаковой длины
        pad_len = self.max_len_out - len(dec_in)
        if pad_len > 0:
            dec_in += [word2idx['<PAD>']] * pad_len
            dec_out += [word2idx['<PAD>']] * pad_len
        else:
            dec_in = dec_in[:self.max_len_out]
            dec_out = dec_out[:self.max_len_out]

        return {
            'enc_in': torch.tensor(enc_in, dtype=torch.long),
            'dec_in': torch.tensor(dec_in, dtype=torch.long),
            'dec_out': torch.tensor(dec_out, dtype=torch.long)
        }

# Параметры последовательностей
max_len_input = 20
max_len_output = 20
dataset = InstructionAnswerDataset(instructions, answers, max_len_input, max_len_output)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# ------------------------------
# 4. Определение модели (Seq2Seq с вниманием)
# ------------------------------

class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers=1, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=word2idx['<PAD>'])
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.hidden_size = hidden_size

    def forward(self, x):
        # x: (batch, seq_len)
        embedded = self.embedding(x)  # (batch, seq_len, embed_size)
        outputs, (hidden, cell) = self.lstm(embedded)
        return outputs, hidden, cell

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attn = nn.Linear(hidden_size * 2, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        # hidden: (batch, hidden_size) - последнее скрытое состояние декодера
        # encoder_outputs: (batch, seq_len, hidden_size)
        batch_size, seq_len, _ = encoder_outputs.size()
        hidden = hidden.unsqueeze(1).repeat(1, seq_len, 1)  # (batch, seq_len, hidden_size)
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))  # (batch, seq_len, hidden_size)
        attention = self.v(energy).squeeze(2)  # (batch, seq_len)
        return torch.softmax(attention, dim=1)

class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers=1, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=word2idx['<PAD>'])
        self.lstm = nn.LSTM(embed_size + hidden_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.attention = Attention(hidden_size)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, prev_hidden, prev_cell, encoder_outputs):
        # x: (batch) - текущий входной токен (индекс)
        # prev_hidden, prev_cell: (num_layers, batch, hidden_size)
        # encoder_outputs: (batch, seq_len, hidden_size)
        x = x.unsqueeze(1)  # (batch, 1)
        embedded = self.dropout(self.embedding(x))  # (batch, 1, embed_size)

        # Вычисляем контекстный вектор через внимание
        hidden_last = prev_hidden[-1]  # (batch, hidden_size) - берём последний слой
        attn_weights = self.attention(hidden_last, encoder_outputs)  # (batch, seq_len)
        attn_weights = attn_weights.unsqueeze(1)  # (batch, 1, seq_len)
        context = torch.bmm(attn_weights, encoder_outputs)  # (batch, 1, hidden_size)

        # Конкатенируем embedding и context
        lstm_input = torch.cat((embedded, context), dim=2)  # (batch, 1, embed_size+hidden_size)

        output, (hidden, cell) = self.lstm(lstm_input, (prev_hidden, prev_cell))
        # output: (batch, 1, hidden_size)
        prediction = self.fc(output.squeeze(1))  # (batch, vocab_size)
        return prediction, hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, enc_in, dec_in, teacher_forcing_ratio=0.5):
        # enc_in: (batch, enc_seq_len)
        # dec_in: (batch, dec_seq_len)
        batch_size = enc_in.size(0)
        dec_seq_len = dec_in.size(1)
        vocab_size = self.decoder.fc.out_features

        encoder_outputs, hidden, cell = self.encoder(enc_in)

        # Первый вход декодера - <SOS> для всех примеров в батче
        decoder_input = dec_in[:, 0]  # (batch)

        outputs = torch.zeros(batch_size, dec_seq_len, vocab_size).to(self.device)

        for t in range(1, dec_seq_len):
            # передаём в декодер предыдущий символ и состояния
            output, hidden, cell = self.decoder(decoder_input, hidden, cell, encoder_outputs)
            outputs[:, t, :] = output

            # teacher forcing: следующий вход либо правильный, либо предсказанный
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)  # (batch)
            decoder_input = dec_in[:, t] if teacher_force else top1

        return outputs

# ------------------------------
# 5. Инициализация модели, функции потерь, оптимизатора
# ------------------------------

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

embed_size = 128
hidden_size = 256
num_layers = 2
dropout = 0.2

encoder = Encoder(vocab_size, embed_size, hidden_size, num_layers, dropout)
decoder = Decoder(vocab_size, embed_size, hidden_size, num_layers, dropout)
model = Seq2Seq(encoder, decoder, device).to(device)

criterion = nn.CrossEntropyLoss(ignore_index=word2idx['<PAD>'])
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ------------------------------
# 6. Обучение
# ------------------------------

num_epochs = 50
for epoch in range(num_epochs):
    total_loss = 0
    model.train()
    for batch in dataloader:
        enc_in = batch['enc_in'].to(device)
        dec_in = batch['dec_in'].to(device)
        dec_out = batch['dec_out'].to(device)

        optimizer.zero_grad()
        output = model(enc_in, dec_in)  # (batch, dec_seq_len, vocab_size)

        # Loss вычисляется только для не-pad токенов
        output = output[:, 1:].reshape(-1, vocab_size)  # пропускаем первый символ (<SOS>)
        dec_out = dec_out[:, 1:].reshape(-1)

        loss = criterion(output, dec_out)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(dataloader):.4f}')

# Сохраняем модель
torch.save(model.state_dict(), 'instruction_answer_model.pth')

# ------------------------------
# 7. Функция для генерации ответа
# ------------------------------

def generate_answer(instruction, max_len=20):
    model.eval()
    with torch.no_grad():
        # кодируем инструкцию
        enc_in = torch.tensor([encode(instruction, max_len_input)]).to(device)  # (1, seq_len)
        encoder_outputs, hidden, cell = model.encoder(enc_in)

        # первый токен декодера - <SOS>
        decoder_input = torch.tensor([word2idx['<SOS>']]).to(device)  # (1)

        answer_ids = []
        for _ in range(max_len):
            output, hidden, cell = model.decoder(decoder_input, hidden, cell, encoder_outputs)
            # output: (1, vocab_size)
            top1 = output.argmax(1).item()
            if top1 == word2idx['<EOS>']:
                break
            answer_ids.append(top1)
            decoder_input = torch.tensor([top1]).to(device)

        return decode(answer_ids)

# Пример использования
print(generate_answer("напиши приветствие"))

print(generate_answer("привет"))

print(generate_answer("Виктор"))

print(generate_answer("Метеориты"))
print(generate_answer("Сколько Жемчужин аксолотля можно найти ежедневно"))
print(generate_answer("Какие бонусы предоставляются после достижения высокого уровня репутации?"))
