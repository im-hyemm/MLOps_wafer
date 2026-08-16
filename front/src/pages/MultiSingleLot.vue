<template>
  <div class="multi-single-lot">
    <div class="container">
      <h1>다중 이미지 분석 (단일 로트)</h1>
      <p>하나의 로트에서 나온 여러 웨이퍼 이미지 zip 파일을 일괄 분석합니다.</p>
      
      <div class="upload-area">
        <input 
          type="file" 
          accept=".zip"
          @change="handleFileUpload"
          class="file-input"
          id="file-input"
        >
        <label for="file-input" class="upload-label">
          <i class="fas fa-cloud-upload-alt"></i>
          <span>ZIP 파일을 선택하세요</span>
        </label>
      </div>

      <div v-if="selectedFile" class="file-list">
        <h3>선택된 파일</h3>
        <div class="files">
          <div class="file-item">
            <span class="file-name">{{ selectedFile.name }}</span>
            <button @click="cancelFileUpload" class="cancel-btn">
              <i class="fas fa-times"></i> 취소
            </button>
          </div>
        </div>
        <div class="action-buttons">
          <button @click="analyzeImages" :disabled="isAnalyzing" class="analyze-btn">
            <i class="fas fa-play"></i> 분석 시작
          </button>
        </div>
      </div>

      <div v-if="isAnalyzing" class="loading">
        <i class="fas fa-spinner fa-spin"></i>
        <span>분석 중...</span>
      </div>

      <!-- 분석 결과 -->
      <div v-if="analysisResult.isGenerated" class="result-section">
        <h3>분석 결과</h3>
        
        <!-- 로트 정보 -->
        <div class="lot-info-section">
          <h5>로트명: {{ analysisResult.lotNumber }}</h5>
        </div>
        
        <!-- 요약 결과 -->
        <div class="summary-section">
          <h5>분석 결과 요약</h5>
          <div class="summary-cards">
            <div class="summary-card total">
              <div class="card-value">{{ analysisResult.totalCount }}</div>
              <div class="card-label">전체 이미지</div>
            </div>
            <div class="summary-card normal">
              <div class="card-value">{{ analysisResult.normalCount }}</div>
              <div class="card-label">정상</div>
            </div>
            <div class="summary-card defective">
              <div class="card-value">{{ analysisResult.defectiveCount }}</div>
              <div class="card-label">불량</div>
            </div>
            <div class="summary-card rate">
              <div class="card-value">{{ (analysisResult.defectiveRate * 100).toFixed(1) }}%</div>
              <div class="card-label">불량률</div>
            </div>
          </div>
        </div>

        <!-- 불량 유형 분석 -->
        <div v-if="analysisResult.defectiveCount > 0" class="defect-analysis-section">
          <h5>불량 유형 분석</h5>
          <div class="donut-chart-container">
            <DonutChart :data="chartDataForDonut" />
            <div class="defect-summary">
              <div class="defect-rate-display">
                <div class="rate-value">{{ (analysisResult.defectiveRate * 100).toFixed(1) }}%</div>
                <div class="rate-label">전체 불량률</div>
              </div>
              <div class="defect-breakdown">
                <h6>불량 유형별 개수</h6>
                <div class="defect-list">
                  <div v-for="(count, type) in analysisResult.defectBreakdown" :key="type" class="defect-item">
                    <span class="defect-dot" :style="{ backgroundColor: getDefectColor(type) }"></span>
                    <span class="defect-name">{{ getDefectTypeName(type) }}</span>
                    <span class="defect-count">{{ count }}개</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>


        <!-- LLM explanation actions -->
        <div v-if="showExplanationActions" class="llm-explanation">
          <button class="llm-button" @click="requestExplanation" :disabled="isRequestingExplanation">
            <i class="fas fa-lightbulb"></i>
            <span>{{ isRequestingExplanation ? 'Requesting...' : 'Request LLM Suggestions' }}</span>
          </button>
          <p v-if="explanationError" class="llm-error">{{ explanationError }}</p>
          <div v-if="explanationText" class="llm-output">
            <h6>LLM Suggestions</h6>
            <div v-html="renderedExplanation"></div>
          </div>
        </div>

        <!-- Process history -->
        <div class="process-history-section">
          <h5>공정 이력</h5>
          <table v-if="analysisResult.processHistory.length" class="process-history-table">
            <thead>
              <tr>
                <th>Step</th>
                <th>Tool</th>
                <th>Recipe</th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="(step, index) in analysisResult.processHistory" :key="`single-lot-process-${index}`">
                <td>{{ step.stepName }}</td>
                <td>{{ step.tool }}</td>
                <td>{{ step.recipe }}</td>
              </tr>
            </tbody>
          </table>
          <p v-else class="process-history-empty">No process history available.</p>
        </div>

        
      </div>

      <!-- 분석 이력 -->
      <div class="history-section">
        <div class="history-header">
          <h3>분석 이력 조회</h3>
          <button @click="toggleHistorySearch" class="search-toggle-btn">
            <i class="fas fa-search"></i> 이력 검색
          </button>
        </div>

        <!-- 검색 필터 -->
        <div v-if="showHistorySearch" class="search-filters">
          <div class="filter-grid">
            <div class="filter-group">
              <label>로트 번호</label>
              <select v-model="searchFilters.lotNumber">
                <option value="">전체</option>
                <option v-for="lot in availableLots" :key="lot" :value="lot">
                  {{ lot }}
                </option>
              </select>
            </div>
            <div class="filter-group">
              <label>분석일시</label>
              <select v-model="searchFilters.analysisDate">
                <option value="">전체</option>
                <option v-for="date in availableDates" :key="date" :value="date">
                  {{ date }}
                </option>
              </select>
            </div>
          </div>
          <div class="filter-actions">
            <button @click="applyFilters" class="apply-btn">검색</button>
            <button @click="resetFilters" class="reset-btn">초기화</button>
          </div>
        </div>

        <div v-if="filteredHistory.length === 0" class="no-history">
          {{ showHistorySearch && hasActiveFilters ? '검색 조건에 맞는 이력이 없습니다.' : '아직 분석 이력이 없습니다.' }}
        </div>
        <div v-else class="history-list">
          <div class="history-count">총 {{ filteredHistory.length }}건의 이력</div>
          <div v-for="(item, index) in filteredHistory" :key="index" class="history-item">
            <div class="history-info">
              <div class="history-meta">
                <span class="session">{{ item.sessionName }}</span>
                <span class="analysis-type">단일 로트</span>
              </div>
              <div class="history-details">
                <p>로트: {{ item.lotNumber }}</p>
                <p>분석일시: {{ item.analysisDate }} | 파일 수: {{ item.fileCount }}개</p>
                <p>결과: 정상 {{ item.results.normal }}개, 불량 {{ item.results.defective }}개</p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
    

  </div>
</template>

<script setup>
import { ref, computed, onMounted } from 'vue'
import DonutChart from '../components/DonutChart.vue'

const selectedFile = ref(null)
const isAnalyzing = ref(false)
const analysisResult = ref({
  isGenerated: false,
  totalCount: 0,
  normalCount: 0,
  defectiveCount: 0,
  defectiveRate: 0,
  defectBreakdown: {},
  lotNumber: '',
  processHistory: []
})
const GLOBAL_STATE_KEY = '__multiSingleLotState__'

// Markdown rendering helpers
const escapeHtml = (value) =>
  value
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;')

const applyInlineMarkdown = (value) =>
  value
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    .replace(/\*(.+?)\*/g, '<em>$1</em>')
    .replace(/`([^`]+)`/g, '<code>$1</code>')

const renderMarkdown = (raw) => {
  if (!raw) {
    return ''
  }

  const lines = raw.split(/\r?\n/)
  const htmlParts = []
  let listBuffer = []

  const flushList = () => {
    if (!listBuffer.length) {
      return
    }
    htmlParts.push(`<ul>${listBuffer.map((item) => `<li>${item}</li>`).join('')}</ul>`)
    listBuffer = []
  }

  lines.forEach((line) => {
    const trimmed = line.trim()

    if (!trimmed) {
      flushList()
      htmlParts.push('<br />')
      return
    }

    const headingMatch = trimmed.match(/^(#{1,6})\s+(.*)$/)
    if (headingMatch) {
      flushList()
      const level = headingMatch[1].length
      const content = applyInlineMarkdown(escapeHtml(headingMatch[2]))
      htmlParts.push(`<h${level}>${content}</h${level}>`)
      return
    }

    if (/^[-*]\s+/.test(trimmed)) {
      const content = applyInlineMarkdown(escapeHtml(trimmed.replace(/^[-*]\s+/, '')))
      listBuffer.push(content)
      return
    }

    flushList()
    const content = applyInlineMarkdown(escapeHtml(trimmed))
    htmlParts.push(`<p>${content}</p>`)
  })

  flushList()
  return htmlParts.join('')
}

const renderedExplanation = computed(() => renderMarkdown(explanationText.value))
const showExplanationActions = computed(
  () => analysisResult.value.isGenerated
)



// Explanation state
const explanationText = ref('')
const explanationError = ref('')
const isRequestingExplanation = ref(false)

const persistAnalysisState = () => {
  if (typeof window === 'undefined') {
    return
  }
  if (!analysisResult.value.isGenerated) {
    window[GLOBAL_STATE_KEY] = null
    return
  }
  window[GLOBAL_STATE_KEY] = {
    analysisResult: JSON.parse(JSON.stringify(analysisResult.value)),
    explanationText: explanationText.value,
    explanationError: explanationError.value,
  }
}

const restoreAnalysisState = () => {
  if (typeof window === 'undefined') {
    return
  }
  const stored = window[GLOBAL_STATE_KEY]
  if (!stored || typeof stored !== 'object') {
    return
  }
  analysisResult.value = {
    ...analysisResult.value,
    ...stored.analysisResult,
    isGenerated: stored.analysisResult?.isGenerated ?? true,
  }
  explanationText.value = stored.explanationText || ''
  explanationError.value = stored.explanationError || ''
}

onMounted(() => {
  restoreAnalysisState()
})

// Explanation helpers
const resetExplanationState = () => {
  explanationText.value = ''
  explanationError.value = ''
  isRequestingExplanation.value = false
  persistAnalysisState()
}

// Build payload for LLM explanation
const buildExplanationPayload = () => {
  const defectCounts = {}

  Object.entries(analysisResult.value.defectBreakdown || {}).forEach(([type, count]) => {
    const numericType = parseInt(type, 10)
    const label = Number.isNaN(numericType) ? type : getDefectTypeName(numericType)
    defectCounts[label] = count
  })

  if (analysisResult.value.normalCount > 0) {
    defectCounts.Normal = analysisResult.value.normalCount
  }

  return {
    lotName: analysisResult.value.lotNumber || 'Unknown lot',
    defectCounts,
    normalCount: analysisResult.value.normalCount,
    defectiveCount: analysisResult.value.defectiveCount,
    defectRate: analysisResult.value.defectiveRate,
  }
}

// Request LLM explanation
const requestExplanation = async () => {
  if (!analysisResult.value.isGenerated) {
    return
  }

  explanationError.value = ''
  explanationText.value = ''
  isRequestingExplanation.value = true

  try {
    const payload = buildExplanationPayload()
    const response = await fetch('http://127.0.0.1:8001/explanation/get_llm_response', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(payload)
    })

    if (!response.ok) {
      let message = 'Failed to retrieve suggestions.'
      try {
        const errorBody = await response.json()
        if (errorBody && errorBody.detail) {
          message = errorBody.detail
        }
      } catch (nestedError) {
        console.warn('Failed to parse error body:', nestedError)
      }
      throw new Error(message)
    }

    const data = await response.json()
    explanationText.value = data.explanation || ''
  } catch (error) {
    console.error('LLM request failed:', error)
    explanationError.value = error.message || 'Failed to retrieve suggestions.'
  } finally {
    isRequestingExplanation.value = false
    persistAnalysisState()
  }
}

// Normalize lot process history for table rendering
const normalizeProcessHistory = (history) => {
  if (!history || typeof history !== 'object') {
    return []
  }

  return Object.entries(history).map(([stepName, details]) => ({
    stepName,
    tool: details && details.tool ? details.tool : '-',
    recipe: details && details.recipe ? details.recipe : '-'
  }))
}

const analysisHistory = ref([
  {
    sessionName: '단일로트분석_2025-01-15',
    lotNumber: 'LOT-2025-001',
    analysisDate: '2025-01-15 15:45:30',
    fileCount: 5,
    results: { normal: 3, defective: 2 }
  },
  {
    sessionName: '단일로트분석_2025-01-14',
    lotNumber: 'LOT-2025-003',
    analysisDate: '2025-01-14 16:30:20',
    fileCount: 7,
    results: { normal: 6, defective: 1 }
  }
])

const showHistorySearch = ref(false)
const searchFilters = ref({
  lotNumber: '',
  analysisDate: ''
})

const getDefectColor = (type) => {
  const defectTypeColors = {
    0: '#f44336',  // Center
    1: '#ff9800',  // Donut
    2: '#9c27b0',  // Edge-Loc
    3: '#2196f3',  // Edge-Ring
    4: '#795548',  // Loc
    5: '#607d8b',  // Random
    6: '#e91e63',  // Scratch
    7: '#ff5722'   // Near-full
  }
  return defectTypeColors[type] || '#f44336'
}

const getDefectTypeName = (type) => {
  const defectTypeNames = {
    0: 'Center',
    1: 'Donut',
    2: 'Edge-Loc',
    3: 'Edge-Ring',
    4: 'Loc',
    5: 'Random',
    6: 'Scratch',
    7: 'Near-full'
  }
  return defectTypeNames[type] || 'Unknown'
}

const chartDataForDonut = computed(() => {
  if (!analysisResult.value.isGenerated || analysisResult.value.totalCount === 0) {
    return { num_by_cat: {} }
  }
  
  const num_by_cat = {}
  
  // 정상 데이터 추가
  if (analysisResult.value.normalCount > 0) {
    num_by_cat['none'] = analysisResult.value.normalCount
  }
  
  // 불량 유형별 데이터 추가
  Object.entries(analysisResult.value.defectBreakdown).forEach(([type, count]) => {
    if (count > 0) {
      const typeNum = parseInt(type)
      num_by_cat[getDefectTypeName(typeNum)] = count
    }
  })
  
  return { num_by_cat }
})

const handleFileUpload = (event) => {
  selectedFile.value = event.target.files[0]
}

const cancelFileUpload = () => {
  selectedFile.value = null
  analysisResult.value.isGenerated = false
  document.getElementById('file-input').value = ''
  // Reset explanation when clearing the selection
  resetExplanationState()
}



const analyzeImages = async () => {
  if (!selectedFile.value) {
    return;
  }

  // Clear previous explanation state for a fresh analysis
  resetExplanationState()
  isAnalyzing.value = true;
  
  const formData = new FormData();
  formData.append('file', selectedFile.value);

  try {
    const response = await fetch('http://127.0.0.1:8001/predict_multi_images_one_lot', {
      method: 'POST',
      body: formData,
    });

    if (response.ok) {
      const data = await response.json();
      const lotProcessHistory = normalizeProcessHistory(data.lotProcessHistory);
      
      const totalCount = data.fileCount;
      const normalCount = data.normal;
      const defectiveCount = data.defective;
      const defectiveRate = data.defectRate;
      
      // 불량 유형별 개수 (백엔드에서 defectTypeCounts로 전달됨)
      const defectBreakdown = {};
      if (data.defectTypeCounts) {
        // 0~7 범위의 불량 유형만 추출
        for (let i = 0; i <= 7; i++) {
          if (data.defectTypeCounts[i] && data.defectTypeCounts[i] > 0) {
            defectBreakdown[i] = data.defectTypeCounts[i];
          }
        }
      }
      
      // 분석 결과 저장
      analysisResult.value = {
        isGenerated: true,
        totalCount,
        normalCount,
        defectiveCount,
        defectiveRate,
        defectBreakdown,
        lotNumber: data.lotNumber || 'Unknown',
        processHistory: lotProcessHistory
      };
      persistAnalysisState();
      
      const historyItem = {
        sessionName: `단일로트분석_${new Date().toLocaleDateString('ko-KR')}`,
        lotNumber: data.lotNumber || 'Unknown',
        analysisDate: new Date().toLocaleString('ko-KR'),
        fileCount: totalCount,
        results: { normal: normalCount, defective: defectiveCount }
      };
      
      analysisHistory.value.unshift(historyItem);
      persistAnalysisState();
    } else {
      console.error('분석 요청 실패:', response.statusText);
      alert('분석에 실패했습니다.');
    }
  } catch (error) {
    console.error('분석 요청 실패:', error);
    alert('분석에 실패했습니다.');
  } finally {
    isAnalyzing.value = false;
  }
}

const availableLots = computed(() => {
  const lots = analysisHistory.value.map(item => item.lotNumber)
  return [...new Set(lots)].sort()
})

const availableDates = computed(() => {
  const dates = analysisHistory.value.map(item => item.analysisDate.split(' ')[0])
  return [...new Set(dates)].sort((a, b) => new Date(b) - new Date(a))
})

const hasActiveFilters = computed(() => {
  return searchFilters.value.lotNumber || searchFilters.value.analysisDate
})

const filteredHistory = computed(() => {
  let filtered = analysisHistory.value
  
  if (searchFilters.value.lotNumber) {
    filtered = filtered.filter(item => item.lotNumber === searchFilters.value.lotNumber)
  }
  
  if (searchFilters.value.analysisDate) {
    filtered = filtered.filter(item => {
      const itemDate = item.analysisDate.split(' ')[0]
      return itemDate === searchFilters.value.analysisDate
    })
  }
  
  return filtered
})

const toggleHistorySearch = () => {
  showHistorySearch.value = !showHistorySearch.value
}

const applyFilters = () => {
  // 필터링은 computed에서 자동으로 처리됨
}

const resetFilters = () => {
  searchFilters.value = {
    lotNumber: '',
    analysisDate: ''
  }
}
</script>

<style scoped>
.multi-single-lot {
  padding: 2rem;
  max-width: 1000px;
  margin: 0 auto;
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
  color: #007bff;
}

.file-list {
  margin: 2rem 0;
}

.files {
  margin: 1rem 0;
}

.file-item {
  padding: 1rem;
  border-bottom: 1px solid #eee;
  display: flex;
  justify-content: space-between;
  align-items: center;
  background: #f8f9fa;
  border-radius: 5px;
  margin-bottom: 1rem;
}

.file-name {
  flex: 1;
  font-weight: 500;
  color: #333;
}

.cancel-btn {
  background: #dc3545;
  color: white;
  border: none;
  padding: 0.5rem 1rem;
  border-radius: 4px;
  cursor: pointer;
  font-size: 0.9rem;
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.cancel-btn:hover {
  background: #c82333;
}

.action-buttons {
  display: flex;
  justify-content: center;
  margin-top: 1rem;
}

.analyze-btn {
  background: #28a745;
  color: white;
  border: none;
  padding: 1rem 2rem;
  border-radius: 5px;
  cursor: pointer;
  font-size: 1rem;
  display: flex;
  align-items: center;
  gap: 0.5rem;
  font-weight: 500;
}

.analyze-btn:hover {
  background: #218838;
}

.analyze-btn:disabled {
  background: #6c757d;
  cursor: not-allowed;
}

.loading {
  text-align: center;
  font-size: 1.2rem;
  color: #007bff;
}

.loading i {
  margin-right: 0.5rem;
}

.history-section {
  margin-top: 3rem;
}

.history-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 1rem;
}

.search-toggle-btn {
  background: #28a745;
  color: white;
  border: none;
  padding: 0.5rem 1rem;
  border-radius: 5px;
  cursor: pointer;
  font-size: 0.9rem;
}

.search-filters {
  background: #f8f9fa;
  padding: 1.5rem;
  border-radius: 10px;
  margin-bottom: 2rem;
  border: 1px solid #e9ecef;
}

.filter-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 1rem;
  margin-bottom: 1rem;
}

.filter-group label {
  display: block;
  margin-bottom: 0.5rem;
  font-weight: 600;
  color: #333;
  font-size: 0.9rem;
}

.filter-group select {
  width: 100%;
  padding: 0.5rem;
  border: 1px solid #ddd;
  border-radius: 5px;
  font-size: 0.9rem;
}

.filter-actions {
  display: flex;
  gap: 1rem;
}

.apply-btn {
  background: #007bff;
  color: white;
  border: none;
  padding: 0.5rem 1.5rem;
  border-radius: 5px;
  cursor: pointer;
}

.reset-btn {
  background: #6c757d;
  color: white;
  border: none;
  padding: 0.5rem 1.5rem;
  border-radius: 5px;
  cursor: pointer;
}

.no-history {
  text-align: center;
  color: #666;
  padding: 2rem;
  background: #f8f9fa;
  border-radius: 10px;
}

.history-list {
  max-height: 400px;
  overflow-y: auto;
}

.history-item {
  background: white;
  border: 1px solid #ddd;
  border-radius: 10px;
  padding: 1.5rem;
  margin-bottom: 1rem;
}

.history-meta {
  display: flex;
  gap: 1rem;
  margin-bottom: 1rem;
  align-items: center;
}

.history-meta span {
  padding: 0.25rem 0.75rem;
  border-radius: 15px;
  font-size: 0.9rem;
  font-weight: 500;
}

.session {
  background: #e8f5e8;
  color: #388e3c;
}

.analysis-type {
  background: #fff3e0;
  color: #f57c00;
}

.history-details p {
  margin: 0.5rem 0;
  color: #666;
}

.history-count {
  margin-bottom: 1rem;
  font-weight: 600;
  color: #666;
}

.result-section {
  background: #f8f9fa;
  padding: 2rem;
  border-radius: 10px;
  margin: 2rem 0;
}

.lot-info-section {
  margin-bottom: 1.5rem;
  padding: 1rem;
  background: #e3f2fd;
  border-radius: 8px;
  border-left: 4px solid #1976d2;
}

.lot-info-section h5 {
  margin: 0;
  color: #1976d2;
  font-weight: 600;
}

.summary-section {
  margin-bottom: 2rem;
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

.defect-analysis-section {
  border-top: 1px solid #e9ecef;
  padding-top: 2rem;
}

.defect-breakdown h6 {
  margin-bottom: 1rem;
  color: #333;
  font-size: 1.1rem;
}

.defect-list {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  gap: 0.75rem;
}

.defect-item {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  padding: 0.75rem;
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
  font-weight: 500;
}

.defect-count {
  font-weight: 600;
  color: #666;
  font-size: 0.9rem;
}

.donut-chart-container {
  display: flex;
  align-items: center;
  gap: 2rem;
  margin-top: 1rem;
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

/* LLM explanation styles */
.llm-explanation {
  margin: 2rem 0;
  display: flex;
  flex-direction: column;
  gap: 1rem;
}

.llm-button {
  align-self: flex-start;
  background: #673ab7;
  color: #ffffff;
  border: none;
  padding: 0.75rem 1.5rem;
  border-radius: 6px;
  cursor: pointer;
  display: inline-flex;
  align-items: center;
  gap: 0.5rem;
  font-weight: 600;
}

.llm-button:disabled {
  opacity: 0.7;
  cursor: not-allowed;
}

.llm-error {
  color: #d32f2f;
  font-weight: 500;
}

.llm-output {
  background: #f3e5f5;
  border-left: 4px solid #673ab7;
  padding: 1rem;
  border-radius: 8px;
  color: #4a148c;
  margin-top: 1rem;
}

.llm-output h6 {
  margin-bottom: 0.5rem;
  font-weight: 700;
}



@media (max-width: 768px) {
  .donut-chart-container {
    flex-direction: column;
    align-items: center;
  }
  
  .defect-list {
    grid-template-columns: 1fr;
  }
}
/* Process history table styling */
.process-history-section {
  margin-top: 2rem;
  border-top: 1px solid #e9ecef;
  padding-top: 2rem;
}

.process-history-table {
  width: 100%;
  border-collapse: collapse;
  background: #ffffff;
  border-radius: 6px;
  overflow: hidden;
  border: 1px solid #dee2e6;
  margin-top: 1rem;
}

.process-history-table th,
.process-history-table td {
  padding: 0.5rem 0.75rem;
  text-align: left;
  border-bottom: 1px solid #e9ecef;
  font-size: 0.9rem;
  color: #444;
}

.process-history-table th {
  background-color: #f1f3f5;
  font-weight: 600;
}

.process-history-table tr:last-child td {
  border-bottom: none;
}

.process-history-empty {
  margin-top: 0.5rem;
  font-size: 0.9rem;
  color: #6c757d;
}

</style>