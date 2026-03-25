# Healthcare Pose Dataset Retrospective

## 1. 배경과 목표

이 작업의 목표는 반려견 측면 보행 영상을 입력받아 헬스케어 지표를 분석하는 서비스에 맞는
커스텀 pose estimation 데이터셋을 직접 설계하고 구축하는 것이었다.

기존 공개 데이터셋을 그대로 사용하는 대신, 실제 서비스 로직에 필요한 keypoint를 재정의하고,
그 스키마에 맞는 라벨 데이터를 만든 뒤 YOLO pose 모델을 학습시켜 보행 분석 품질을 높이려 했다.

핵심 목표는 다음과 같았다.

- 측면 보행 영상 전용 keypoint schema 설계
- CVAT를 이용한 수동 라벨링 파이프라인 구축
- 소량 수동 라벨을 이용한 1차 모델 학습
- 학습된 모델로 나머지 데이터 자동 라벨링
- 보정 라벨을 누적해 점진적으로 데이터셋 확장

## 2. 이번 작업에서 실제로 한 일

이번 작업에서는 다음 단계를 수행했다.

1. AI-Hub 반려견 보행 데이터에서 `Left/Right` 측면 프레임만 추출했다.
2. 서비스용 `healthcare-side-final-23` keypoint schema를 정의했다.
3. seed dataset과 CVAT import용 zip을 생성했다.
4. CVAT에서 813장 task를 만들고, 그중 앞부분 151장을 수동 보정했다.
5. 수동 보정한 151장으로 1차 YOLO pose 모델을 fine-tuning 했다.
6. 학습 결과물로 [for151.pt](/root/medical_AI/hkh/jacob/ktb_fp_healthcare/models/for151.pt)를 만들었다.
7. 이 모델로 남은 662장에 자동 라벨링을 시도했다.
8. 자동 라벨 결과를 다시 CVAT 보정용 데이터셋으로 만들려 했으나, 결과 품질이 낮아 중단했다.

관련 자료:

- 스키마 제안서: [HEALTHCARE_SIDE_SCHEMA_PROPOSAL.md](/root/medical_AI/hkh/jacob/ktb_fp_healthcare/docs/HEALTHCARE_SIDE_SCHEMA_PROPOSAL.md)
- 1차 학습 결과: [results.csv](/root/medical_AI/hkh/jacob/ktb_fp_healthcare/models/for151_lowmem_e100_run/train_yolo26s-pose/results.csv)
- 1차 모델: [for151.pt](/root/medical_AI/hkh/jacob/ktb_fp_healthcare/models/for151.pt)

## 3. 결과 요약

결론적으로 이번 시도는 **23개 keypoint full schema를 기준으로 한 semi-automatic dataset expansion에는 실패**했다고 보는 것이 맞다.

이유는 다음과 같다.

- 151장만으로는 23개 keypoint를 안정적으로 학습시키기 어려웠다.
- 측면 영상에서는 구조적으로 잘 보이지 않는 far-side proximal point가 많았다.
- 1차 모델이 약한 상태에서 auto-label을 확장하자 invisible point hallucination이 많이 발생했다.
- 반대편 점이 가까운 쪽에 겹쳐 찍히는 `near/far collapse` 문제가 크게 나타났다.

즉, 파이프라인은 구현됐지만 현재 스키마와 데이터 조건 조합으로는 usable한 2차 자동 라벨 결과를 얻지 못했다.

## 4. 실패 원인 분석

### 4-1. 스키마가 관측 조건보다 컸다

처음 정의한 23점 스키마는 서비스 입장에서는 매력적이었지만, 실제 측면 보행 영상에서는 모든 점이 안정적으로 관측되지 않았다.

특히 아래 점들은 구조적으로 자주 안 보였다.

- `far_front_shoulder`
- `far_front_elbow`
- `far_rear_hip`
- `far_rear_stifle`
- `sacrum`
- 일부 `tail` 계열

즉 "라벨링이 어렵다"가 아니라, 영상 조건 자체가 이 점들을 지속적으로 제공하지 못했다.

### 4-2. 수동 라벨 데이터 양이 부족했다

151장은 bootstrap 용도로는 의미 있었지만, 23개 점 전체를 학습시키기엔 부족했다.

더 큰 문제는 총 장수보다 **실제로 각 keypoint가 관측된 예시 수**가 적었다는 점이다.
far-side와 tail/sacrum 계열은 151장 안에서도 supervision이 매우 희소했다.

### 4-3. 1차 모델을 너무 이른 시점에 자동 라벨 확장에 사용했다

1차 모델은 최종 모델이 아니라 bootstrap 모델이어야 하는데, 이번에는 그 bootstrap 성능이 아직 부족했다.

결과적으로 자동 라벨에서 다음 문제가 나타났다.

- invisible point hallucination
- near/far collapse
- 반대쪽 점을 가까운 쪽 점과 같은 위치에 복제
- 코, far-side paw, tail point의 위치 불안정

### 4-4. pose estimation 자체의 난도가 높았다

이번 작업은 bbox object detection이 아니라 pose estimation keypoint annotation이었다.
이 차이가 난도를 크게 올렸다.

pose keypoint 작업에서는 다음을 모두 맞춰야 한다.

- 점의 해부학적 의미
- 점 순서
- 가시성 판단
- near/far 구분
- occlusion 처리
- 프레임 간 일관성

특히 측면 영상에서 반대편 점은 "존재는 하지만 확실히 보이지 않는" 애매한 경우가 많아서 라벨 품질 관리가 훨씬 어렵다.

## 5. 이번 작업의 장점

이번 시도가 실패로 끝났다고 해도, 얻은 것이 분명히 있었다.

- 실제 서비스 요구사항에서 출발해 custom schema를 설계해봤다.
- AI-Hub 원본 데이터, seed dataset, CVAT, YOLO 학습, auto-label 확장까지 전체 파이프라인을 경험했다.
- 데이터 품질이 모델 성능을 얼마나 직접적으로 좌우하는지 확인했다.
- "서비스가 원한다고 해서 영상이 그 정보를 항상 제공하는 것은 아니다"라는 점을 실전으로 확인했다.
- 단순 학습이 아니라 dataset engineering 관점의 문제를 경험했다.

## 6. 이번 작업의 단점

- 초기에 스키마를 너무 크게 잡아 라벨링 비용과 모델 난이도를 동시에 키웠다.
- 관측 가능성 검증보다 full schema 구현을 먼저 시도했다.
- 1차 모델의 신뢰도보다 큰 범위에 auto-label을 걸어 오히려 후속 보정 부담이 커졌다.
- 결과적으로 자동 라벨이 사람 작업을 줄이기보다, 다시 사람이 확인해야 할 오류를 늘렸다.

## 7. 포트폴리오 관점에서 어떻게 정리할 수 있는가

이 작업은 "완성된 모델을 만들었다"는 성과보다, **문제 정의와 실패 분석을 통해 데이터 설계의 본질을 배웠다**는 관점으로 정리하는 것이 적절하다.

포트폴리오에서는 다음 흐름으로 설명할 수 있다.

1. 서비스 요구사항 분석
2. keypoint schema 설계
3. CVAT 기반 수동 라벨링 파이프라인 구성
4. 1차 수동 라벨로 pose model bootstrap
5. auto-label 확장 시도
6. 실패 패턴 분석
7. observability 기반 schema pruning 필요성 도출

즉 이번 경험은 "모델링 성공 사례"라기보다 **dataset design, labeling strategy, observability mismatch를 다뤄본 사례**로 충분히 의미가 있다.

## 8. bbox object detection과 비교했을 때

이번 실패 원인을 설명할 때, "bbox였다면 더 쉬웠다"는 판단은 대체로 맞다.

object detection은 다음 특성이 있다.

- 객체 위치만 대략적으로 잡으면 됨
- occlusion이 있어도 박스 수준에서 처리가 가능함
- near/far 같은 해부학적 분리가 필요 없음
- annotation 속도가 훨씬 빠름

반면 pose estimation은:

- 부위별 정확한 위치를 알아야 함
- 안 보이는 점은 비워야 함
- 보이는 점만 일관되게 찍어야 함
- 좌우/near/far를 구분해야 함
- 잘못된 point 하나가 skeleton 전체 품질을 무너뜨릴 수 있음

즉 이번 작업은 단순 bbox보다 훨씬 정교하고 고난도인 문제였고, 그만큼 첫 시도에서 실패한 것도 충분히 설명 가능하다.

## 9. 다음에 다시 한다면 어떻게 하는 게 좋은가

다시 한다면 아래 순서로 가는 것이 훨씬 현실적이다.

### 9-1. 스키마를 먼저 줄인다

23점 full schema가 아니라, 측면 영상에서 실제로 자주 관측되는 core point만 남긴다.

예시:

- `withers`
- `t13_spinous_process`
- `iliac_crest`
- near-side limb chain
- 일부 far-side distal point
- 필요 시 `nose`

즉 "서비스가 원하면 다 넣는다"가 아니라, **실제 영상에서 안정적으로 보이는가**를 기준으로 schema를 줄여야 한다.

### 9-2. observability 검증을 먼저 한다

100~200장을 먼저 수동 라벨링하면서 각 point의 관측률을 계산하고,
그 결과로 유지/제거를 결정해야 한다.

### 9-3. 그 다음 수동 라벨 수를 늘린다

151장은 bootstrap으로는 시작점이지만, 23점 pose를 안정적으로 확장하기엔 부족했다.
다시 한다면 더 많은 수동 라벨이 필요하다.

### 9-4. auto-label은 보수적으로 쓴다

- 기존 seed 유지
- 높은 confidence threshold 사용
- 일부 신뢰 가능한 point만 자동 fill
- 전부 덮어쓰기 금지

### 9-5. 목적을 분리한다

최종 서비스용 분석 지표와 오버레이 가독성을 동시에 한 번에 해결하려 하지 말고,

- 분석에 꼭 필요한 point
- 오버레이 시각화에 있으면 좋은 point

를 분리해서 단계적으로 확장하는 편이 낫다.

## 10. 최종 정리

이번 작업은 결과적으로 목표한 23점 pose dataset 확장에는 실패했다.
하지만 그 실패는 단순한 시행착오가 아니라, 다음 사실을 분명히 보여줬다.

- 서비스 요구사항과 영상 관측 가능성은 다를 수 있다.
- keypoint dataset 설계는 모델보다 먼저 검증돼야 한다.
- pose estimation annotation은 bbox detection보다 훨씬 고난도다.
- 적은 수동 라벨로 큰 스키마를 곧바로 semi-auto 확장하는 것은 위험하다.

따라서 이번 경험은 실패 사례라기보다,
**데이터셋 설계와 라벨링 전략을 실제로 시행하고 한계를 분석해본 고급 실험 경험**으로 정리하는 것이 가장 적절하다.
