# GPT 
Responsible for generating GPT calls
- **A_llamarec**: `llamarec` preference and perception generation  
- **input**: Text input of `embeding.py`  
- **embeding.py**: Performs `embedding` on the text information from `input` and maps it to the input dimension of `SASRec`  
- **SASRec_emb**: Calls GPT to generate text content embedded in `SASRec`  

# LlamaRec
 Training and inference of LlamaRec
- **0_Grocery_and_Gourmet_Food_sft0.py**: Constructs the GPT call prompt  
- **0_Grocery_and_Gourmet_Food_sft1.py**: Builds SFT training data and trains using `Llama-Factory`  
- **0_sft_data.py**: Directly constructs SFT without perception  
- **2_inference.py**: Performs inference on the trained model  
- **bias.py**: Computes bias  

# LlamaRec_plus:
LlamaRec training and inference enhanced with SASRec
- Structure is the same as **LlamaRec**  

# llmehance
 SASRec plus training and inference
- **SASRec_item_emb**  
  - Adds `item embedding`  
  - `pretrain`: Stores offline generated `item embedding`  
  - `run.py`: Performs training  
- **SASRec_user_emb**
  - Adds `user embedding`
  - `pretrain`: Stores offline generated `user embedding` and `item embedding`
  - `run.py`: Performs training  


# BERT4Rec results
## Table 1: Results of CRM-as-Retriever.

| User Embedding | Item Embedding | MIND Hit | MIND NDCG | Food Hit | Food NDCG | Yelp Hit | Yelp NDCG |
|---------------|---------------|----------|-----------|----------|-----------|----------|-----------|
| None          | Random        | 0.1440   | 0.0760    | 0.0193   | 0.0099    | 0.0305   | 0.0156    |
| None          | Caption       | 0.1513   | 0.0799    | 0.0270   | 0.0149    | 0.0316   | 0.0157    |
| None          | Description   | 0.1561   | 0.0829    | 0.0289   | 0.0154    | 0.0327   | 0.0160    |
| Random        | Caption       | 0.1447   | 0.0757    | 0.0288   | 0.0155    | 0.0316   | 0.0162    |
| Random        | Description   | 0.1460   | 0.0783    | 0.0295   | 0.0161    | 0.0331   | 0.0166    |
| Preference    | Caption       | 0.1604   | 0.0865    | 0.0308   | 0.0161    | 0.0361   | 0.0175    |
| Preference    | Description   | 0.1609   | 0.0868    | 0.0310   | 0.0162    | 0.0367   | 0.0181    |


## Table 2: Results of LLM-as-Ranker.

| Retriever | Ranker  | MIND NDCG | MIND MAPB | Food NDCG | Food MAPB | Yelp NDCG | Yelp MAPB |
|-----------|--------|-----------|-----------|-----------|-----------|-----------|-----------|
| CRM       | None   | 0.0760    | -         | 0.0099    | -         | 0.0156    | -         |
| CRM       | LLM    | 0.0798    | 0.9553    | 0.0113    | 0.9172    | 0.0157    | 1.3938    |
| CRM       | LLM+   | 0.0799    | 0.8948    | 0.0117    | 0.9076    | 0.0157    | 1.1010    |
| CRM       | LLM++  | 0.0801    | 0.8259    | 0.0118    | 0.7895    | 0.0159    | 1.0860    |
| CRM++     | None   | 0.0868    | -         | 0.0162    | -         | 0.0181    | -         |
| CRM++     | LLM    | 0.0869    | 1.0052    | 0.0180    | 1.1181    | 0.0184    | 1.4115    |
| CRM++     | LLM+   | 0.0872    | 0.8933    | 0.0182    | 0.8337    | 0.0184    | 1.2870    |
| CRM++     | LLM++  | 0.0878    | 0.8878    | 0.0183    | 0.7820    | 0.0185    | 1.2407    |
