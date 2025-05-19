This is the PyTorch implementation of the proposed model LAGCL4Rec.
<br>

<p align='center'>
<img src="./backbone.png" width="282" height="156"><br>
<i>Figure: The Framework of LAGCL4Rec</i>
</p>

<br>
We developed our code in the following environment:
<br>

```
Python == 3.12
torch == 2.5.1
```

We provide the ml-1m example dataset and XSimGCL implementation code. To execute this code:

1. First you need to disable reranking
2. Train the rank model
3. Output cf_candidates.json
4. Download the model (Qwen2.5 7b instruct) to the ./rerank/model directory
5. Train the Qwen model
6. After the Qwen model training is complete, you can use LAGCL4Rec normally