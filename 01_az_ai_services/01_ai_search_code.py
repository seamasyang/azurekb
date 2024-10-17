import logging
logging.basicConfig(level=logging.WARNING)


from FlagEmbedding import BGEM3FlagModel


model_name = "BAAI/bge-m3"
embedding_model = BGEM3FlagModel(
    model_name, use_fp16=True, normalize_embeddings=False
)

print("start to embedding:")
vector = embedding_model.encode("济南如何领取社保卡", return_sparse=True)
lac = embedding_model.convert_id_to_token(vector["lexical_weights"])

print(f"dense vector = {vector["dense_vecs"]}")
print(f"lac = {lac}")