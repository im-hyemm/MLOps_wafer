<template>
  <div class="pkl-prediction">
    <div class="container">
      <h1>PKL 데이터 분석</h1>
      <p>PKL 형식의 데이터를 업로드하여 분석합니다.</p>

      <!-- 분석 탭 -->
      <div v-if="activeTab === 'analysis'" class="tab-content">
      
      <div class="upload-area">
        <input 
          type="file" 
          accept=".pkl" 
          @change="handleFileUpload"
          class="file-input"
          id="pkl-input"
        >
        <label for="pkl-input" class="upload-label">
          <i class="fas fa-database"></i>
          <span>PKL 파일을 선택하세요</span>
        </label>
      </div>

      <div v-if="selectedFile" class="preview-result">
        <h3>선택된 파일</h3>
        <div class="file-details">
          <p><strong>파일명:</strong> {{ selectedFile.name }}</p>
          <p><strong>크기:</strong> {{ formatFileSize(selectedFile.size) }}</p>
        </div>
        <button @click="predictLabeledPickle" class="analyze-btn">분석 시작</button>
      </div>

      <div v-if="isAnalyzing" class="loading">
        <i class="fas fa-spinner fa-spin"></i>
        <span>데이터 분석 중...</span>
      </div>

      <!-- .pkl 데이터 분석 결과 -->
      <div v-if="result" class="preview-result">
        <!-- 데이터 특성 분석 -->
        <div class="data-characteristics">
          <h3>데이터 특성 분석</h3>
          <div class="characteristics-grid">
            <!-- 데이터 크기 card -->
            <div class="char-card">
              <div class="char-icon">
                <i class="fas fa-table"></i>
              </div>
              <div class="char-info">
                <h6>데이터 크기</h6>
                <p>{{ result.meta.nrow }}개 레코드</p>
              </div>
            </div>

            <!-- Lot 개수 card -->
            <div class="char-card">
              <div class="char-icon">
                <i class="fas fa-columns"></i>
              </div>
              <div class="char-info">
                <h6>Lot 개수</h6>
                <p>{{ result.meta.nlot }}개</p>
              </div>
            </div>

            <!-- 데이터 품질 card -->
            <div class="char-card">
              <div class="char-icon">
                <i class="fas fa-chart-line"></i>
              </div>
              <div class="char-info">
                <h6>데이터 품질</h6>
                <p>{{ result.meta.labeled_ratio }}%</p>
              </div>
            </div>
          </div>
        </div>
      </div>

      <!-- 실제 데이터 통계 도너트 차트 -->
      <div v-if="result" class="preview-result">
        <h3>실제 데이터 통계</h3>

        <div class="donut-chart-container">
          <DonutChart :data="result.meta" />
          
          <!-- 통계 요약 -->
          <div class="defect-summary">
            <div class="defect-rate-display">
              <div class="rate-value">{{ result.meta.defect_ratio }}%</div>
              <div class="rate-label">전체 불량률</div>
            </div>

            <div class="defect-breakdown">
              <h6>불량 유형별 개수</h6>
              <div class="defect-list">
                <div v-for="(count, type) in result.meta.num_by_cat" :key="type" class="defect-item">
                  <span class="defect-dot" :style="{ backgroundColor: getDefectColor(type) }"></span>
                  <span class="defect-name">{{ type }}</span>
                  <span class="defect-count">{{ count }}개</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      <!-- 실제 데이터 요약 결과 -->
      <div v-if="result" class="preview-result">
        <h3>실제 데이터 요약</h3>
        <div class="summary-cards">
          <div class="summary-card total">
            <div class="card-value">{{ result.meta.nrow }}</div>
            <div class="card-label">전체 데이터</div>
          </div>
          <div class="summary-card normal">
            <div class="card-value">{{ result.meta.num_by_cat.none }}</div>
            <div class="card-label">정상</div>
          </div>
          <div class="summary-card defective">
            <div class="card-value">{{ result.meta.nrow - result.meta.num_by_cat.none }}</div>
            <div class="card-label">불량</div>
          </div>
          <div class="summary-card rate">
            <div class="card-value">{{ result.meta.defect_ratio }}%</div>
            <div class="card-label">불량률</div>
          </div>
        </div>
      </div>

      <!-- 기존 모델 기반 성능 지표 -->
      <div v-if="result" class="preview-result">
        <h3>모델 성능 지표</h3>
        <div class="metrics-grid">
          <div class="metric-card" :class="getMetricCardClass(getCurrentValue('overal_acc') * 100, 75)">
            <div class="metric-value">
              {{ (getCurrentValue('overal_acc') * 100).toFixed(1) }}%
              <span v-if="retrainResponse?.new && getMetricChange('overal_acc')" 
                    :class="['change-indicator', getMetricChange('overal_acc').class]">
                {{ getMetricChange('overal_acc').icon }} {{ getMetricChange('overal_acc').text }}
              </span>
            </div>
            <div class="metric-label">정확도</div>
          </div>
          <div class="metric-card" :class="getMetricCardClass(getCurrentValue('macro_precision'), 0.7)">
            <div class="metric-value">
              {{ getCurrentValue('macro_precision').toFixed(3) }}
              <span v-if="retrainResponse?.new && getMetricChange('macro_precision')" 
                    :class="['change-indicator', getMetricChange('macro_precision').class]">
                {{ getMetricChange('macro_precision').icon }} {{ getMetricChange('macro_precision').text }}
              </span>
            </div>
            <div class="metric-label">정밀도</div>
          </div>
          <div class="metric-card" :class="getMetricCardClass(getCurrentValue('macro_recall'), 0.7)">
            <div class="metric-value">
              {{ getCurrentValue('macro_recall').toFixed(3) }}
              <span v-if="retrainResponse?.new && getMetricChange('macro_recall')" 
                    :class="['change-indicator', getMetricChange('macro_recall').class]">
                {{ getMetricChange('macro_recall').icon }} {{ getMetricChange('macro_recall').text }}
              </span>
            </div>
            <div class="metric-label">재현율</div>
          </div>
          <div class="metric-card" :class="getMetricCardClass(getCurrentValue('macro_f1'), 0.7)">
            <div class="metric-value">
              {{ getCurrentValue('macro_f1').toFixed(3) }}
              <span v-if="retrainResponse?.new && getMetricChange('macro_f1')" 
                    :class="['change-indicator', getMetricChange('macro_f1').class]">
                {{ getMetricChange('macro_f1').icon }} {{ getMetricChange('macro_f1').text }}
              </span>
            </div>
            <div class="metric-label">F1 Score</div>
          </div>
        </div>

        <!-- 모델 재학습 추천 -->
        <div class="retraining-recommendation">
          <div class="recommendation-card" :class="retrainingRecommendation.status">
            <div class="recommendation-icon">
              <i :class="retrainingRecommendation.icon"></i>
            </div>
            <div class="recommendation-content">
              <h6>{{ retrainingRecommendation.title }}</h6>
              <p>{{ retrainingRecommendation.message }}</p>
              <div class="recommendation-details">
                <ul>
                  <li v-for="reason in retrainingRecommendation.reasons" :key="reason">{{ reason }}</li>
                </ul>
              </div>
              <!-- 모델 재학습 버튼 -->
              <button @click="retrainModel" :disabled="isRetraining" class="retrain-btn">
                <i :class="isRetraining ? 'fas fa-spinner fa-spin' : 'fas fa-sync-alt'"></i>
                {{ isRetraining ? '재학습 진행 중...' : '모델 재학습 시작' }}
              </button>
            </div>
          </div>
        </div>

      </div>

      <!-- 모델 기반 성능 지표 -->
      <div v-if="result" class="preview-result">
        <h3>모델 성능 지표 세부 분석</h3>
        <!-- 오류 유형별 모델 성능 지표 -->
        <table class="performance-metrics-table">
          <thead>
            <tr>
              <th>오류 유형</th>
              <th>Precision</th>
              <th>Recall</th>
              <th>F1 Score</th>
              <th>Support</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="(metrics, index) in getCurrentMetrics()" :key="index">
              <td>{{ metrics.label }}</td>
              <td>
                {{ metrics.precision }}
                <span v-if="retrainResponse?.new && getDetailChange(metrics, 'precision')" 
                      :class="['change-indicator', getDetailChange(metrics, 'precision')?.class]">
                  {{ getDetailChange(metrics, 'precision')?.icon }} {{ getDetailChange(metrics, 'precision')?.text }}
                </span>
              </td>
              <td>
                {{ metrics.recall }}
                <span v-if="retrainResponse?.new && getDetailChange(metrics, 'recall')" 
                      :class="['change-indicator', getDetailChange(metrics, 'recall')?.class]">
                  {{ getDetailChange(metrics, 'recall')?.icon }} {{ getDetailChange(metrics, 'recall')?.text }}
                </span>
              </td>
              <td>
                {{ metrics.f1 }}
                <span v-if="retrainResponse?.new && getDetailChange(metrics, 'f1')" 
                      :class="['change-indicator', getDetailChange(metrics, 'f1')?.class]">
                  {{ getDetailChange(metrics, 'f1')?.icon }} {{ getDetailChange(metrics, 'f1')?.text }}
                </span>
              </td>
              <td>{{ metrics.support }}</td>
            </tr>
          </tbody>
        </table>

        <br>
        <div class="preview-note">
          <i class="fas fa-info-circle"></i>
          <span>이는 모델 추론 결과이며, 실제 데이터 결과와 상이할 수 있습니다.</span>
        </div>
      </div>

    </div>
    <!-- 모델 버전  -->
    <!-- <ModelStatus :status="modelStatus" /> -->
    </div>
  </div>
</template>

<script setup>
import { ref, computed } from 'vue';
import axios from 'axios';

// 컴포넌트 import
import ModelStatus from '../components/ModelStatus.vue';
import DonutChart from '../components/DonutChart.vue'

const selectedFile = ref(null);
const isAnalyzing = ref(false);
const isRetraining = ref(false);
const activeTab = ref('analysis');
const result = ref(null); // 서버에서 받은 응답
const retrainResponse = ref(null); // 재학습 응답
const errorMessage = ref('');

// 차트 색상 변수
const DEFECT_TYPE_COLORS = {
  'Center': '#f44336',
  'Donut': '#ff9800',
  'Edge-Loc': '#9c27b0',
  'Edge-Ring': '#2196f3',
  'Loc': '#795548',
  'Random': '#607d8b',
  'Scratch': '#e91e63',
  'Near-full': '#ff5722',
  'none': '#4caf50', // 정상
}

// 모델 버전 관련 dummy
const modelStatus = ref({
  version: '1.2.0',
  lastTrained: '2025-09-08',
  updateLogs: [
    { date: '2025-09-08', description: '새로운 유형의 불량 데이터로 모델 재학습 완료' },
    { date: '2025-09-01', description: '초기 모델 배포' },
  ],
})

const handleFileUpload = (event) => {
  const f = event.target.files[0];
  if (f) {
    selectedFile.value = f;
    result.value = null;
    errorMessage.value = '';
  } else {
    selectedFile.value = null;
  }
}

// 분석 시작 버튼 클릭 시, labeled data 분석 결과 api 요청해서 응답 받음
const predictLabeledPickle = async () => {
  isAnalyzing.value = true;
  result.value = null;
  errorMessage.value = '';
  const formData = new FormData();
  formData.append('file', selectedFile.value);

  try {
    const response = await axios.post('http://127.0.0.1:8001/upload_predict_labeled_images', formData,
      { headers: {
        'Content-Type': 'multipart/form-data'
      },
    });
    result.value = response.data;
    // console.log(result.value);
    selectedFile.value = null; // 업로드 후 파일 초기화
  } catch (error) {
    if (axios.isAxiosError(error) && error.response) {
      errorMessage.value = `오류: ${error.response.status} - ${error.response.statusText}`;
    } else {
      errorMessage.value = (error && error.message) || "알 수 없는 오류가 발생했습니다.";
    }
  } finally {
    isAnalyzing.value = false;
  }
}

const getDefectColor = (type) => {
    return DEFECT_TYPE_COLORS[type] || '#ccc';
}

const retrainingRecommendation = computed(() => {
  if (!result.value) return {};

  const accuracy = parseFloat(result.value.used.overal_acc) * 100;
  const precision = parseFloat(result.value.used.macro_precision);
  const recall = parseFloat(result.value.used.macro_recall);
  const f1Score = parseFloat(result.value.used.macro_f1);
  
  const reasons = [];
  let needsRetraining = false;

  if (accuracy < 75) {
    needsRetraining = true;
    reasons.push(`정확도가 ${accuracy.toFixed(2)}%로 낮음 (기준: 75% 이상)`);
  }
  if (precision < 0.7) {
    needsRetraining = true;
    reasons.push(`정밀도가 ${precision.toFixed(2)}%로 낮음 (기준: 0.7 이상)`);
  }
  if (recall < 0.7) {
    needsRetraining = true;
    reasons.push(`재현률이 ${recall.toFixed(2)}%로 낮음 (기준: 0.7 이상)`);
  }
  if (f1Score < 0.7) {
    needsRetraining = true;
    reasons.push(`F1 Score가 ${f1Score.toFixed(2)}로 낮음 (기준: 0.7 이상)`);
  }

  if (needsRetraining) {
    return {
      status: 'needs-retraining',
      icon: 'fas fa-exclamation-triangle',
      title: '모델 재학습 추천',
      message: '성능 지표가 기준치를 하회하여 모델 재학습을 추천합니다.',
      reasons
    };
  } else {
    return {
      status: 'good-performance',
      icon: 'fas fa-check-circle',
      title: '모델 성능 양호',
      message: '모든 성능 지표가 기준치를 충족하여 재학습이 필요하지 않습니다.',
      reasons: ['모든 성능 지표가 기준치 이상', '데이터 품질 양호', '안정적인 모델 성능']
    };
  }
});

const formatFileSize = (bytes) => {
  if (bytes === 0) return '0 Bytes'
  const k = 1024
  const sizes = ['Bytes', 'KB', 'MB', 'GB']
  const i = Math.floor(Math.log(bytes) / Math.log(k))
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i]
}

// 재학습 요청 함수
const retrainModel = async () => {
  // 결과가 존재하고 file_location이 있는지 확인
  if (!result.value || !result.value.file_location) {
    console.error('File location not found.');
    return;
  }

  isRetraining.value = true;

  // 재학습 요청 시 보내는 데이터
  const dataToSend = {
    file_location: result.value.file_location,
    f1_score: parseFloat(result.value.used.macro_f1),
  };

  try {
    // axios를 사용해 POST 요청 보내기
    const response = await axios.post('http://127.0.0.1:8001/retrain_predict_labeled_images', dataToSend);
    // 요청이 성공한 경우
    console.log('재학습 요청 성공:', response.data);
    
    // 재학습 응답 저장
    retrainResponse.value = response.data;
    
    if (response.data.model_change) {
      // 변화량 계산을 위해 원본 데이터 백업
      if (response.data.new) {
        retrainResponse.value.originalAggregates = {
          overal_acc: result.value.used.overal_acc,
          macro_precision: result.value.used.macro_precision,
          macro_recall: result.value.used.macro_recall,
          macro_f1: result.value.used.macro_f1,
        };
        retrainResponse.value.originalMetrics = JSON.parse(JSON.stringify(result.value.used.metrics_by_cat));
        
        // 기존 값을 새로운 값으로 업데이트
        result.value.used.overal_acc = response.data.new.overal_acc;
        result.value.used.macro_precision = response.data.new.macro_precision;
        result.value.used.macro_recall = response.data.new.macro_recall;
        result.value.used.macro_f1 = response.data.new.macro_f1;
        result.value.used.metrics_by_cat = response.data.new.metrics_by_cat;
      }
      alert('모델 재학습이 완료되었습니다. 재학습된 모델로 교체합니다.');
    } else {
      alert('재학습 모델 성능 미흡: 기존 모델을 유지합니다.');
    }
  } catch (error) {
    // 요청 실패한 경우
    if (axios.isAxiosError(error) && error.response) {
      console.error(`오류: ${error.response.status} - ${error.response.statusText}`);
    } else {
      console.error((error && error.message) || "알 수 없는 오류가 발생했습니다.");
    }
    alert('모델 재학습 요청에 실패했습니다.');
  } finally {
    isRetraining.value = false;
  }
};

// Metric change helpers
const metricKeys = ['overal_acc', 'macro_precision', 'macro_recall', 'macro_f1'];

const metricChangeMap = computed(() => {
  if (!retrainResponse.value?.originalAggregates || !retrainResponse.value?.new) {
    return {};
  }
  const changes = {};
  metricKeys.forEach((key) => {
    const base = parseFloat(retrainResponse.value.originalAggregates[key]);
    const updated = parseFloat(retrainResponse.value.new[key]);
    if (Number.isNaN(base) || Number.isNaN(updated)) {
      changes[key] = null;
      return;
    }
    const diff = parseFloat((updated - base).toFixed(3));
    const magnitude = Math.abs(diff).toFixed(3);
    if (Math.abs(diff) < 0.001) {
      changes[key] = { icon: '−', class: 'change-none', text: magnitude };
    } else if (diff > 0) {
      changes[key] = { icon: '▲', class: 'change-up', text: magnitude };
    } else {
      changes[key] = { icon: '▼', class: 'change-down', text: magnitude };
    }
  });
  return changes;
});

const getMetricChange = (key) => metricChangeMap.value[key];

const getDetailChangeDisplay = (oldMetrics, newMetrics, key) => {
  if (!oldMetrics || !newMetrics) return null;
  const base = parseFloat(oldMetrics[key]);
  const updated = parseFloat(newMetrics[key]);
  if (Number.isNaN(base) || Number.isNaN(updated)) return null;
  const diff = parseFloat((updated - base).toFixed(3));
  const magnitude = Math.abs(diff).toFixed(3);
  if (Math.abs(diff) < 0.001) return { icon: '−', class: 'change-none', text: magnitude };
  if (diff > 0) return { icon: '▲', class: 'change-up', text: magnitude };
  return { icon: '▼', class: 'change-down', text: magnitude };
};

const getDetailChange = (metrics, key) => {
  if (!retrainResponse.value?.originalMetrics) return null;
  const original = findOriginalMetrics(metrics.label);
  if (!original) return null;
  return getDetailChangeDisplay(original, metrics, key);
};

const getCurrentValue = (key) => {
  return result.value?.used[key] || 0;
};

// 현재 메트릭 반환
const getCurrentMetrics = () => {
  return result.value?.used?.metrics_by_cat || [];
};

// 새로운 메트릭 찾기
const findNewMetrics = (label) => {
  if (!retrainResponse.value?.new?.metrics_by_cat) return null;
  return Object.values(retrainResponse.value.new.metrics_by_cat).find(m => m.label === label);
};

// 원래 메트릭 찾기 (변화량 계산용)
const findOriginalMetrics = (label) => {
  if (!retrainResponse.value?.originalMetrics) return null;
  return Object.values(retrainResponse.value.originalMetrics).find(m => m.label === label);
};

// 성능 지표 값에 따라 css 변화하는 함수
const getMetricCardClass = (value, threshold) => {
    return value >= threshold ? 'metric-good' : 'metric-bad';
}

</script>

<style scoped>
.pkl-prediction {
  padding: 2rem;
  max-width: 1000px;
  margin: 0 auto;
}

.tab-navigation {
  display: flex;
  margin: 2rem 0 1rem 0;
  border-bottom: 2px solid #e9ecef;
}

.tab-button {
  background: none;
  border: none;
  padding: 1rem 2rem;
  cursor: pointer;
  font-size: 1rem;
  font-weight: 500;
  color: #666;
  border-bottom: 3px solid transparent;
  transition: all 0.3s ease;
}

.tab-button:hover {
  color: #28a745;
  background: #f8f9fa;
}

.tab-button.active {
  color: #28a745;
  border-bottom-color: #28a745;
  font-weight: 600;
}

.tab-content {
  margin-top: 2rem;
}

.preview-section {
  background: #f8f9fa;
  padding: 2rem;
  border-radius: 10px;
}

.preview-form {
  background: white;
  padding: 1.5rem;
  border-radius: 10px;
  margin: 1.5rem 0;
  border: 1px solid #e9ecef;
}

.auto-analysis-info {
  margin-bottom: 1.5rem;
}

.info-card {
  display: flex;
  align-items: center;
  gap: 1rem;
  padding: 1.5rem;
  background: #e8f5e8;
  border: 1px solid #c3e6c3;
  border-radius: 10px;
}

.info-card i {
  font-size: 2rem;
  color: #28a745;
}

.info-text h6 {
  margin: 0 0 0.5rem 0;
  color: #28a745;
  font-size: 1rem;
}

.info-text p {
  margin: 0;
  color: #155724;
  font-size: 0.9rem;
  line-height: 1.4;
}

.preview-btn {
  background: #28a745;
  color: white;
  border: none;
  padding: 1rem 2rem;
  border-radius: 5px;
  cursor: pointer;
  font-size: 1rem;
}

.preview-result {
  background: white;
  padding: 2rem;
  border-radius: 10px;
  margin-top: 2rem;
  border: 1px solid #e9ecef;
}

.data-characteristics {
  margin-bottom: 2rem;
}

.characteristics-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 1rem;
  margin-top: 1rem;
}

.char-card {
  display: flex;
  align-items: center;
  gap: 1rem;
  padding: 1.5rem;
  background: #f8f9fa;
  border-radius: 10px;
  border: 1px solid #e9ecef;
}

.char-icon {
  font-size: 2rem;
  color: #28a745;
}

.char-info h6 {
  margin: 0 0 0.5rem 0;
  color: #333;
  font-size: 1rem;
}

.char-info p {
  margin: 0;
  color: #666;
  font-weight: 600;
}

.performance-metrics {
  margin-bottom: 2rem;
}

.retraining-recommendation {
  position: relative;
  margin: 2rem 0;
}

.recommendation-card {
  position: relative;
  display: flex;
  align-items: flex-start;
  gap: 1.5rem;
  padding: 2rem;
  border-radius: 10px;
  border: 2px solid;
}

.pkl-prediction .retraining-recommendation .recommendation-card {
  transition: background 0.3s ease, border-color 0.3s ease, color 0.3s ease;
}

.pkl-prediction .retraining-recommendation .recommendation-card.needs-retraining {
  background: linear-gradient(135deg, #e53e3e 0%, #ff6b35 100%);
  border-color: #e53e3e;
  color: #ffffff;
}

.pkl-prediction .retraining-recommendation .recommendation-card.good-performance {
  background: linear-gradient(135deg, #1d976c 0%, #31c77b 100%);
  border-color: #1d976c;
  color: #ffffff;
}

.recommendation-icon {
  font-size: 2.5rem;
  flex-shrink: 0;
}

.recommendation-content h6 {
  margin: 0 0 0.5rem 0;
  font-size: 1.2rem;
  font-weight: 600;
}

.recommendation-content p {
  /* margin: 0 0 1rem 0; */
  line-height: 1.5;
}

.recommendation-details ul {
  margin: 0;
  padding-left: 1.5rem;
}

.recommendation-details li {
  margin-bottom: 0.5rem;
  font-size: 0.9rem;
}

.metrics-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: 1rem;
  margin-top: 1rem;
}

.metric-card {
  text-align: center;
  padding: 1.5rem;
  border-radius: 10px;
}

.metric-good {
  background: #e8f5e8;
  border: 2px solid #388e3c;
  color: #388e3c;
}

.metric-bad {
  background: #ffebee;
  border: 2px solid #d32f2f;
  color: #d32f2f;
}

.metric-value {
  font-size: 2rem;
  font-weight: bold;
  margin-bottom: 0.5rem;
}

.metric-label {
  font-size: 0.9rem;
  font-weight: 500;
}

.defect-analysis-section {
  margin-bottom: 2rem;
  border-top: 1px solid #e9ecef;
  padding-top: 2rem;
}

.donut-chart-container {
  display: flex;
  align-items: center;
  gap: 2rem;
  margin-top: 1rem;
}

.donut-chart {
  flex-shrink: 0;
}

.donut-label {
  font-size: 12px;
  font-weight: 600;
  fill: #333;
}

.defect-summary {
  flex: 1;
}

.defect-rate-display {
  text-align: center;
  margin-bottom: 2rem;
  padding: 1.5rem;
  background: #f8f9fa;
  border-radius: 10px;
}

.rate-value {
  font-size: 3rem;
  font-weight: bold;
  color: #d32f2f;
  margin-bottom: 0.5rem;
}

.rate-label {
  font-size: 1.1rem;
  color: #666;
  font-weight: 500;
}

.defect-breakdown h6 {
  margin-bottom: 1rem;
  color: #333;
  font-size: 1.1rem;
}

.defect-list {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 0.75rem;
}

.defect-item {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  padding: 0.5rem;
  background: white;
  border-radius: 6px;
  border: 1px solid #e9ecef;
}

.defect-dot {
  width: 12px;
  height: 12px;
  border-radius: 50%;
  flex-shrink: 0;
}

.defect-name {
  flex: 1;
  font-size: 0.9rem;
  color: #333;
}

.defect-count {
  font-weight: 600;
  color: #666;
  font-size: 0.9rem;
}

.preview-summary {
  border-top: 1px solid #e9ecef;
  padding-top: 2rem;
}

.summary-cards {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: 1rem;
  margin: 1rem 0;
}

.summary-card {
  text-align: center;
  padding: 1.5rem;
  border-radius: 10px;
  border: 2px solid;
}

.summary-card.total {
  background: #e3f2fd;
  border-color: #1976d2;
  color: #1976d2;
}

.summary-card.normal {
  background: #e8f5e8;
  border-color: #388e3c;
  color: #388e3c;
}

.summary-card.defective {
  background: #ffebee;
  border-color: #d32f2f;
  color: #d32f2f;
}

.summary-card.rate {
  background: #fff3e0;
  border-color: #f57c00;
  color: #f57c00;
}

.card-value {
  font-size: 2rem;
  font-weight: bold;
  margin-bottom: 0.5rem;
}

.card-label {
  font-size: 0.9rem;
  font-weight: 500;
}

.preview-details {
  margin-top: 1rem;
}

.preview-details p {
  margin: 0.5rem 0;
  color: #333;
}

.preview-note {
  display: flex;
  align-items: center;
  gap: 2rem;
  margin-top: 0.5rem;
  padding: 1rem;
  background: #fff3cd;
  border: 1px solid #ffeaa7;
  border-radius: 5px;
  color: #856404;
  font-size: 0.9rem;
}

@media (max-width: 768px) {
  .characteristics-grid {
    grid-template-columns: 1fr;
  }
  
  .metrics-grid {
    grid-template-columns: repeat(2, 1fr);
  }
  
  .donut-chart-container {
    flex-direction: column;
    align-items: center;
  }
  
  .defect-list {
    grid-template-columns: 1fr;
  }
  
  .summary-cards {
    grid-template-columns: repeat(2, 1fr);
  }
  
  .recommendation-card {
    flex-direction: column;
    text-align: center;
  }
}

.upload-area {
  border: 2px dashed #ccc;
  border-radius: 10px;
  padding: 2rem;
  text-align: center;
  margin: 2rem 0;
}

.file-input {
  display: none;
}

.upload-label {
  cursor: pointer;
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 1rem;
  font-size: 1.2rem;
  color: #666;
}

.upload-label i {
  font-size: 3rem;
  color: #28a745;
}

.file-info {
  margin: 2rem 0;
  padding: 1rem;
  border: 1px solid #ddd;
  border-radius: 5px;
}

.file-details p {
  margin: 0.5rem 0;
}

.analyze-btn {
  background: #28a745;
  color: white;
  border: none;
  padding: 1rem 2rem;
  border-radius: 5px;
  cursor: pointer;
  font-size: 1rem;
  margin-top: 1rem;
}

.loading {
  text-align: center;
  font-size: 1.2rem;
  color: #28a745;
}

.loading i {
  margin-right: 0.5rem;
}

.performance-metrics-table {
  width: 100%;
  border-collapse: collapse;
  margin-top: 20px;
}

.performance-metrics-table th,
.performance-metrics-table td {
  padding: 10px;
  text-align: center;
  border: 1px solid #ddd;
}

.performance-metrics-table th {
  background-color: #f4f4f4;
}

.performance-metrics-table tr:nth-child(even) {
  background-color: #f9f9f9;
}

.performance-metrics-table td {
  font-size: 14px;
}

.retrain-btn {
  text-align: center;
  padding: 1.5rem;
  border-radius: 10px;
  background: #28a745;
  border: 3px solid #1e7e34;
  color: white;
  font-size: 1.1rem;
  font-weight: bold;
  cursor: pointer;
  transition: all 0.3s ease;
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 0.5rem;
  box-shadow: 0 4px 12px rgba(40, 167, 69, 0.3);
  animation: pulse 2s infinite;

  /* 버튼을 추천 카드 내에서 우측 하단으로 배치 */
  position: absolute;
  bottom: 20px; /* 카드 하단에서 20px 떨어짐 */
  right: 20px;  /* 카드 우측에서 20px 떨어짐 */
}

@keyframes pulse {
  0% {
    box-shadow: 0 4px 12px rgba(40, 167, 69, 0.3);
  }
  50% {
    box-shadow: 0 6px 20px rgba(40, 167, 69, 0.5);
  }
  100% {
    box-shadow: 0 4px 12px rgba(40, 167, 69, 0.3);
  }
}

.retrain-btn:hover:not(:disabled) {
  transform: translateY(-2px);
  box-shadow: 0 5px 15px rgba(0,0,0,0.2);
}

.retrain-btn:disabled {
  opacity: 0.7;
  cursor: not-allowed;
  transform: none;
}

.change-indicator {
  display: block;
  font-size: 0.7rem;
  font-weight: normal;
  margin-top: 0.2rem;
}

.change-up {
  color: #28a745;
}

.change-down {
  color: #dc3545;
}

.change-none {
  color: #6c757d;
}

</style>




