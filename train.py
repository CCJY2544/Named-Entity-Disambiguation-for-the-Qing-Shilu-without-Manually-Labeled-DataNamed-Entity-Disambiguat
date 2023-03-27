import torch
from transformers import BertTokenizer
import random
import os
import pandas as pd
from tqdm import tqdm, trange
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import BertForSequenceClassification
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

class FakeNewsDataset(Dataset):
    # 讀取前處理後的 tsv 檔並初始化一些參數
    def __init__(self, mode, tokenizer):
        assert mode in ["train", "test", "dev"]  # 一般訓練你會需要 DevSet
        self.mode = mode
        self.df = pd.read_csv("ming_"+mode + ".tsv", sep="\t").fillna("")
        self.len = len(self.df)
        self.label_map = {'Y': 0, 'N': 1}
        self.tokenizer = tokenizer  # 我們使用 BERT tokenizer
    
    # 定義回傳一筆訓練 / 測試數據的函式
    def __getitem__(self, idx):
        if self.mode == "test":
            text_a, text_b = self.df.iloc[idx, :2].values
            label_tensor = None
        else:
            text_a, text_b, label = self.df.iloc[idx, :].values
            # 將 label 文字也轉換成索引方便轉換成 tensor
            label_id = self.label_map[label]
            label_tensor = torch.tensor(label_id)
            
        # 建立第一個句子的 BERT tokens 並加入分隔符號 [SEP]
        word_pieces = ["[CLS]"]
        tokens_a = self.tokenizer.tokenize(text_a)
        word_pieces += tokens_a + ["[SEP]"]
        len_a = len(word_pieces)
        
        # 第二個句子的 BERT tokens
        tokens_b = self.tokenizer.tokenize(text_b)
        word_pieces += tokens_b + ["[SEP]"]
        len_b = len(word_pieces) - len_a
        
        # 將整個 token 序列轉換成索引序列
        ids = self.tokenizer.convert_tokens_to_ids(word_pieces)
        tokens_tensor = torch.tensor(ids)
        
        # 將第一句包含 [SEP] 的 token 位置設為 0，其他為 1 表示第二句
        segments_tensor = torch.tensor([0] * len_a + [1] * len_b, dtype=torch.long)
        
        return (tokens_tensor, segments_tensor, label_tensor)
    
    def __len__(self):
        return self.len
    

def create_mini_batch(samples):
    tokens_tensors = [s[0] for s in samples]
    segments_tensors = [s[1] for s in samples]
    
    # 測試集有 labels
    if samples[0][2] is not None:
        label_ids = torch.stack([s[2] for s in samples])
    else:
        label_ids = None
    
    # zero pad 到同一序列長度
    tokens_tensors = pad_sequence( tokens_tensors, batch_first=True)
    segments_tensors = pad_sequence( segments_tensors, batch_first=True)
    
    # attention masks，將 tokens_tensors 裡頭不為 zero padding 的位置設為 1 ，讓 BERT 只關注這些位置的 tokens
    masks_tensors = torch.zeros( tokens_tensors.shape, dtype=torch.long)
    masks_tensors = masks_tensors.masked_fill( tokens_tensors != 0, 1)
    
    return tokens_tensors, segments_tensors, masks_tensors, label_ids

###算分數
def simple_accuracy(preds, labels):
    # logger.info("preds")
    # logger.info(preds)
    # logger.info("labels")
    # logger.info(labels)
    return (preds == labels).mean()
def acc_and_f1(preds, labels):
    #acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds,average='macro')
    return {
        "f1": f1
    }

def get_predictions(model, dataloader, compute_acc=False):
    y_pred = []
    y_true = []
    predictions = None
    correct = 0
    total = 0
    with torch.no_grad():
        for data in dataloader:
            # 將所有 tensors 移到 GPU 上
            if next(model.parameters()).is_cuda:
                data = [t.to("cuda:1") for t in data if t is not None]
            
            
            # 別忘記前 3 個 tensors 分別為 tokens, segments 以及 masks ， 且強烈建議在將這些 tensors 丟入 model 時指定對應的參數名稱
            tokens_tensors, segments_tensors, masks_tensors = data[:3]
            outputs = model(input_ids=tokens_tensors, 
                            token_type_ids=segments_tensors, 
                            attention_mask=masks_tensors)
            
            logits = outputs[0]
            _, pred = torch.max(logits.data, 1)
            
            labels = data[3]

            for ele in pred:
                y_pred.append(ele.item())
            for element in labels:
                #print(element)
                y_true.append(element.item())
            
            # 用來計算訓練集的分類準確率
            if compute_acc:
                labels = data[3]
                total += labels.size(0)
                correct += (pred == labels).sum().item()
            
            # 將當前 batch 記錄下來
            if predictions is None:
                predictions = pred
            else:
                predictions = torch.cat((predictions, pred))
    if compute_acc:
        acc = correct / total
        print(acc_and_f1(y_pred, y_true))
        return predictions,acc

def training(mode, BATCH_SIZE, NUM_LABELS, PRETRAINED_MODEL_NAME, GPU, EPOCHS, output_file, record_file):
    # 初始化一個專門讀取訓練樣本的 Dataset，使用中文 BERT 斷詞
    trainset = FakeNewsDataset(mode, tokenizer=tokenizer)

    # 初始化一個每次回傳 x 個訓練樣本的 DataLoader 利用 `collate_fn` 將 list of samples 合併成一個 mini-batch 是關鍵
    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, collate_fn=create_mini_batch)

    model = BertForSequenceClassification.from_pretrained(PRETRAINED_MODEL_NAME, num_labels=NUM_LABELS)

    # high-level 顯示此模型裡的 modules
    print("""name            module
    ----------------------""")
    for name, module in model.named_children():
        if name == "bert":
            for n, _ in module.named_children():
                print(f"{name}:{n}")
        else:
            print("{:15} {}".format(name, module))

    # 讓模型跑在 GPU 上並取得訓練集的分類準確率
    device = torch.device(GPU if torch.cuda.is_available() else "cpu")
    print("device:", device)
    model = model.to(device)

    # 訓練模式
    model.train()
    # 使用 Adam Optim 更新整個分類模型的參數
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    # 紀錄 acc loss f1-score
    txt = open(record_file, "a", encoding="utf8")

    for epoch in trange(EPOCHS):
        print(epoch)
        running_loss = 0.0
        for data in trainloader:
            torch.cuda.empty_cache()
            tokens_tensors, segments_tensors, \
            masks_tensors, labels = [t.to(device) for t in data]
            # 將參數梯度歸零
            optimizer.zero_grad()
            
            # forward pass
            outputs = model(input_ids=tokens_tensors, 
                            token_type_ids=segments_tensors, 
                            attention_mask=masks_tensors, 
                            labels=labels)

            loss = outputs[0]
            # backward
            loss.backward()
            optimizer.step()

            # 紀錄當前 batch loss
            running_loss += loss.item()

            
            
        # 計算分類準確率
        _, acc = get_predictions(model, trainloader, compute_acc=True)

        print('[epoch %d] loss: %.3f, acc: %.3f' %
            (epoch + 1, running_loss, acc))
        txt.writelines( '[epoch %d] loss: %.3f, acc: %.3f\n' %(epoch + 1, running_loss, acc) )
    output_dir = output_file
    torch.save({
                'epoch': EPOCHS,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss}, "./checkpoint")
    model.save_pretrained(output_dir)

    

PRETRAINED_MODEL_NAME = "bert-base-chinese"  
tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"

# input
df_train = pd.read_csv("./data/BaseUnused_train.csv")
df_train = df_train.sample(frac=1, random_state=42)

# 去除不必要的欄位，重新命名欄位名

df_train = df_train.reset_index()
df_train = df_train.loc[:, ['text_a', 'text_b', 'label']]
df_train.columns = ['text_a', 'text_b', 'label']

# 將處理結果另存成 tsv 
df_train.to_csv("./data/BaseUnused_train.tsv", sep="\t", index=False)

###########
training(mode = "train", BATCH_SIZE = 2, NUM_LABELS = 2, PRETRAINED_MODEL_NAME = "bert-base-chinese", GPU = "cuda:1",\
        EPOCHS = 20, output_file = './BaseUnused15-2-20/', record_file ='./BaseUnused15-2-20/record.txt')



#Dev
print("\n-------------------I am dev")

df_dev = pd.read_csv("./data/BaseUnused_dev.csv")
output_dir = './BaseUnused15-4-20(BaseUnused-token)/'
model = BertForSequenceClassification.from_pretrained(output_dir)
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

df_dev = df_dev.loc[:, ["text_a", "text_b", "label"]]
df_dev.columns = ["text_a", "text_b", "label"]

df_dev.to_csv("./data/BaseUnused_dev.tsv", sep="\t", index=False)
devset = FakeNewsDataset("dev", tokenizer=tokenizer)
devloader = DataLoader(devset, batch_size=12, collate_fn=create_mini_batch)

predictions,acc = get_predictions(model, devloader, compute_acc=True)
print("Accuracy : "+str(acc))




#Test
print("\n-------------------I am test")

df_test = pd.read_csv("./data/ming_test.csv")
output_dir = './QQQ15-8-20(Baseline)/'
model = BertForSequenceClassification.from_pretrained(output_dir)
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

df_test = df_test.loc[:, ["text_a", "text_b", "label"]]
df_test.columns = ["text_a", "text_b", "label"]

df_test.to_csv("./data/ming_test.tsv", sep="\t", index=False)
testset = FakeNewsDataset("test", tokenizer=tokenizer)
testloader = DataLoader(testset, batch_size=2, collate_fn=create_mini_batch)

predictions,acc = get_predictions(model, testloader, compute_acc=True)
print("Accuracy : "+str(acc))



# 用來將預測的 label id 轉回 label 文字
index_map = {v: k for k, v in devset.label_map.items()}
print("\n-------------生成 csv 檔案")
# 生成 Kaggle 繳交檔案
df = pd.DataFrame({"Category": predictions.tolist()})
df['Category'] = df.Category.apply(lambda x: index_map[x])
print(devset.df)
df_pred = pd.concat([df_dev.loc[:, ["text_a"]],df_dev.loc[:, ["text_b"]],df_dev.loc[:, ["label"]], df.loc[:, 'Category']], axis=1)
# df_pred = pd.concat([devset.df], df.loc[:, 'Category'], axis=1)

df_pred.to_csv('result.csv', index=False)
df_pred.head()

