from sentence_transformers import SentenceTransformer, losses, models, InputExample
from torch.utils.data import DataLoader
import os

def fine_tune(df, output_dir='models/fine_tuned_sales_embed'):
    examples = []
    for _, row in df.iterrows():
        for chunk in row['chunks']:
            label = float(row['label'])
            examples.append(InputExample(texts=[chunk, chunk], label=label))

    word_embedding_model = models.Transformer('sentence-transformers/all-MiniLM-L6-v2')
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

    train_dataloader = DataLoader(examples, shuffle=True, batch_size=16)
    train_loss = losses.CosineSimilarityLoss(model=model)

    model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=3, warmup_steps=100)
    os.makedirs(output_dir, exist_ok=True)
    model.save(output_dir)