import torch
from torch_geometric.nn import GCNConv
from torch_geometric.nn.aggr import SortAggregation as SortPool
from torch.nn import Conv1d, MaxPool1d, Linear, Dropout
from torch_geometric.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score

class DGCNN(torch.nn.Module):
    def __init__(self, dim_in, k=30):
        """
        DGCNN 모델을 초기화합니다.
        :param dim_in: 노드 특성의 입력 차원
        :param k: 글로벌 정렬 풀링 시 선택할 k 값
        """
        super().__init__()

        # GCN 레이어 정의: 입력 -> 32 -> 32 -> 32 -> 1 차원으로 축소
        self.gcn1 = GCNConv(dim_in, 32)
        self.gcn2 = GCNConv(32, 32)
        self.gcn3 = GCNConv(32, 32)
        self.gcn4 = GCNConv(32, 1)

        # 글로벌 정렬 풀링 (SortPool) 레이어
        self.global_pool = SortPool(k=k)

        # SortPool 출력 후 시계열 Conv1d + MaxPool1d 블록
        # 1채널 -> 16채널, 필터 크기 97, 스트라이드 97
        self.conv1 = Conv1d(in_channels=1, out_channels=16, kernel_size=97, stride=97)
        # 16채널 -> 32채널, 필터 크기 5, 스트라이드 1
        self.conv2 = Conv1d(in_channels=16, out_channels=32, kernel_size=5, stride=1)
        # 맥스풀링 (윈도우 크기 2, 스트라이드 2)
        self.maxpool = MaxPool1d(kernel_size=2, stride=2)

        # Fully connected 레이어: Conv 출력 -> 128 유닛
        # SortPool (k*채널) 크기를 352로 가정
        self.linear1 = Linear(352, 128)
        # 드롭아웃 (확률 0.5)
        self.dropout = Dropout(0.5)
        # 최종 출력 레이어: 128 -> 1 (이진 분류)
        self.linear2 = Linear(128, 1)

    def forward(self, x, edge_index, batch):
        """
        순전파 함수
        :param x: 노드 특성 행렬 [총_노드, dim_in]
        :param edge_index: 엣지 인덱스 텐서
        :param batch: 배치 정보 (각 노드가 어느 그래프에 속하는지)
        :return: 그래프당 1차원 시그모이드 출력
        """
        # 1. GCN으로 노드 임베딩 추출 및 tanh 활성화
        h1 = self.gcn1(x, edge_index).tanh()
        h2 = self.gcn2(h1, edge_index).tanh()
        h3 = self.gcn3(h2, edge_index).tanh()
        h4 = self.gcn4(h3, edge_index).tanh()

        # 2. 각 레이어 출력들을 채널 차원으로 concat
        h = torch.cat([h1, h2, h3, h4], dim=-1)

        # 3. 글로벌 정렬 풀링 적용 -> 각 그래프당 [batch_size, k*1] 크기
        h = self.global_pool(h, batch)

        # 4. Conv1d를 위해 차원 재정렬: [batch, 채널=1, 길이]
        h = h.view(h.size(0), 1, h.size(-1))
        # 5. Conv1d + 활성화
        h = self.conv1(h).relu()
        # 6. MaxPool1d
        h = self.maxpool(h)
        # 7. 두 번째 Conv1d + 활성화
        h = self.conv2(h).relu()

        # 8. Flatten: [batch, 채널*길이]
        h = h.view(h.size(0), -1)
        # 9. Fully connected + 활성화
        h = self.linear1(h).relu()
        # 10. 드롭아웃
        h = self.dropout(h)
        # 11. 최종 레이어 & 시그모이드 출력
        h = self.linear2(h).sigmoid()

        return h

# GPU 사용 설정 (가능 시)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 모델 생성 및 디바이스 배치
dim_in = train_dataset[0].num_features
model = DGCNN(dim_in).to(device)
# 옵티마이저 & 학습률 설정
optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-4)
# 손실 함수: BCEWithLogitsLoss (시그모이드 포함)
criterion = torch.nn.BCEWithLogitsLoss()

# 학습 함수 정의
def train():
    model.train()
    total_loss = 0.0

    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch)
        # 예측값과 레이블 간 손실 계산
        loss = criterion(out.view(-1), data.y.to(torch.float))
        loss.backward()
        optimizer.step()
        # 배치 손실 누적 (그래프 수 가중치)
        total_loss += float(loss) * data.num_graphs

    # 평균 손실 반환
    return total_loss / len(train_dataset)

# 평가 함수 정의
def test(loader):
    model.eval()
    y_pred, y_true = [], []

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data.x, data.edge_index, data.batch)
            y_pred.append(out.view(-1).cpu())
            y_true.append(data.y.view(-1).cpu().to(torch.float))

    # ROC AUC 및 Average Precision 계산
    auc = roc_auc_score(torch.cat(y_true), torch.cat(y_pred))
    ap = average_precision_score(torch.cat(y_true), torch.cat(y_pred))

    return auc, ap

# 학습 루프 실행 및 검증 지표 출력
for epoch in range(31):
    loss = train()
    val_auc, val_ap = test(val_loader)
    print(f'Epoch {epoch:>2} | Loss: {loss:.4f} | Val AUC: {val_auc:.4f} | Val AP: {val_ap:.4f}')

# 테스트 세트 평가
test_auc, test_ap = test(test_loader)
print(f'Test AUC: {test_auc:.4f} | Test AP {test_ap:.4f}')
