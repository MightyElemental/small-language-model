import pandas as pd
import torch
from torch.utils.data import Dataset
from datetime import timedelta

class DiscordDataset(Dataset):
    def __init__(self, csv_file, time_threshold=300, tokenizer=None, vocab=None, max_seq_length=512):
        """
        Args:
            csv_file (str): Path to the CSV file with discord messages.
            time_threshold (int): Time gap (in seconds) to separate conversations.
            tokenizer (callable): Function to split text into tokens; defaults to whitespace split.
            vocab (dict): Optional vocabulary mapping tokens to indices; if None, one will be built.
            max_seq_length (int): Maximum sequence length.
        """
        self.data = pd.read_csv(csv_file)
        # Ensure the 'time' column is parsed as datetime
        self.data['time'] = pd.to_datetime(self.data['time'])
        self.data.sort_values(by='time', inplace=True)
        self.time_threshold = timedelta(seconds=time_threshold)
        self.tokenizer = tokenizer if tokenizer is not None else lambda x: x.split()
        self.max_seq_length = max_seq_length

        # Group messages into conversations based on time gaps.
        self.conversations = self.group_conversations()

        # Build or use provided vocabulary.
        if vocab is None:
            self.vocab = self.build_vocab(self.conversations)
        else:
            self.vocab = vocab

        # Process conversations into training samples.
        self.samples = self.process_conversations()

    def group_conversations(self):
        conversations = []
        current_conv = []
        prev_time = None

        for _, row in self.data.iterrows():
            current_time = row['time']
            if prev_time is not None and (current_time - prev_time) > self.time_threshold:
                if current_conv:
                    conversations.append(current_conv)
                current_conv = []
            current_conv.append(row)
            prev_time = current_time
        if current_conv:
            conversations.append(current_conv)
        return conversations

    def build_vocab(self, conversations):
        from collections import Counter
        counter = Counter()
        # Define special tokens. Assume index 0 is <pad>.
        special_tokens = ['<pad>', '<unk>', '<bos>', '<eos>']
        for conv in conversations:
            # For simplicity, only use conversations with at least two participants.
            usernames = list({row['username'] for row in conv})
            if len(usernames) < 2:
                continue
            for row in conv:
                tokens = self.tokenizer(row['message'])
                counter.update(tokens)
        vocab = {token: idx for idx, token in enumerate(special_tokens)}
        for token, _ in counter.items():
            if token not in vocab:
                vocab[token] = len(vocab)
        return vocab

    def process_conversations(self):
        samples = []
        for conv in self.conversations:
            # Map multiple usernames into two roles.
            unique_users = list({row['username'] for row in conv})
            if len(unique_users) < 2:
                continue
            # Assign the first encountered user as USER and the second as ASSISTANT.
            mapping = {}
            mapping[unique_users[0]] = 'USER'
            if len(unique_users) > 1:
                mapping[unique_users[1]] = 'ASSISTANT'
            # Any additional users are mapped to ASSISTANT by default.
            for user in unique_users[2:]:
                mapping[user] = 'ASSISTANT'

            # Build a conversation string with role tags.
            conversation_text = ""
            for row in conv:
                role = mapping.get(row['username'], 'USER')
                # Format: <bos> ROLE: message <eos>
                conversation_text += f"<bos> {role}: {row['message']} <eos> "
            tokens = self.tokenizer(conversation_text)
            # Convert tokens to indices (use <unk> for unknown tokens).
            token_ids = [self.vocab.get(token, self.vocab['<unk>']) for token in tokens]
            # Truncate to max sequence length.
            if len(token_ids) > self.max_seq_length:
                token_ids = token_ids[:self.max_seq_length]
            # For language modeling, create input (all tokens except last) and target (all tokens except first).
            if len(token_ids) < 2:
                continue
            input_ids = token_ids[:-1]
            target_ids = token_ids[1:]
            samples.append((torch.tensor(input_ids, dtype=torch.long),
                            torch.tensor(target_ids, dtype=torch.long)))
        return samples

    def collate_fn(self, batch):
        """
        Pads a batch of (input_ids, target_ids) pairs to the same length.
        """
        from torch.nn.utils.rnn import pad_sequence
        input_ids, target_ids = zip(*batch)
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.vocab['<pad>'])
        target_ids = pad_sequence(target_ids, batch_first=True, padding_value=self.vocab['<pad>'])
        return input_ids, target_ids

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

# For conversation data, consider using publicly available datasets.
# For example:
#  - Cornell Movie Dialogs Corpus: https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html
#  - Persona-Chat Dataset: https://github.com/facebookresearch/ParlAI/tree/main/projects/convai2
# These datasets provide conversational exchanges that can be used for training chat models.
