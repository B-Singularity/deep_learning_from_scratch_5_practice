import os
import numpy as np


def multivariate_normal(xs, mu, cov):
    """벡터화된 다변수 정규 분포 확률 밀도 함수"""
    d = xs.shape[1]  # 특성 차원
    det = np.linalg.det(cov)
    inv = np.linalg.inv(cov)
    z = 1 / np.sqrt((2 * np.pi) ** d * det)

    diffs = xs - mu  # (N, D)
    exps = np.einsum('nd,dd,nd->n', diffs, inv, diffs)  # Mahalanobis 거리 벡터화 계산
    return z * np.exp(-0.5 * exps)  # (N,) 형태 결과 반환


def gmm(xs, phis, mus, covs):
    """벡터화된 GMM 확률 밀도 계산"""
    K = len(phis)
    ys = np.array([
        phis[k] * multivariate_normal(xs, mus[k], covs[k])
        for k in range(K)
    ])  # (K, N) 형태
    return np.sum(ys, axis=0)  # (N,) 형태


def likelihood(xs, phis, mus, covs):
    """로그 우도 계산"""
    eps = 1e-8
    y = gmm(xs, phis, mus, covs)
    return np.mean(np.log(y + eps))


def e_step(xs, phis, mus, covs):
    """벡터화된 E-step"""
    K = len(phis)
    qs = np.array([
        phis[k] * multivariate_normal(xs, mus[k], covs[k])
        for k in range(K)
    ]).T  # (N, K) 형태 변환
    qs /= np.sum(qs, axis=1, keepdims=True)  # 정규화
    return qs


def m_step(xs, qs):
    """벡터화된 M-step"""
    N, K = qs.shape
    qs_sum = qs.sum(axis=0)  # (K,)

    # phis 업데이트
    phis = qs_sum / N

    # mus 업데이트 (벡터 연산)
    mus = (qs.T @ xs) / qs_sum[:, np.newaxis]  # (K, D)

    # covs 업데이트 (배치 연산)
    diffs = xs[:, np.newaxis, :] - mus  # (N, K, D)
    covs = np.einsum('nk,nkd,nkj->kdj', qs, diffs, diffs) / qs_sum[:, np.newaxis, np.newaxis]  # (K, D, D)

    return phis, mus, covs


# 데이터 로드
path = os.path.join(os.path.dirname(__file__), 'old_faithful.txt')
xs = np.loadtxt(path)

# 초기화
phis = np.array([0.5, 0.5])
mus = np.array([[0.0, 50.0],
                [0.0, 100.0]])
covs = np.array([np.eye(2), np.eye(2)])

# EM 알고리즘 실행
MAX_ITERS = 100
THRESHOLD = 1e-4
current_likelihood = likelihood(xs, phis, mus, covs)

for _ in range(MAX_ITERS):
    qs = e_step(xs, phis, mus, covs)
    phis, mus, covs = m_step(xs, qs)

    next_likelihood = likelihood(xs, phis, mus, covs)
    print(f'{next_likelihood:.3f}')
    if np.abs(current_likelihood - next_likelihood) < THRESHOLD:
        break
    current_likelihood = next_likelihood