<template>
  <div class="multi-prediction">
    <div class="container">
      <h1>다중 이미지 분석 (여러 로트)</h1>
      <p>여러 로트에서 나온 웨이퍼 이미지 zip 파일을 일괄 분석합니다.</p>
      
      <!-- 분석 탭 -->
      <div class="tab-content">

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
        
        
        <!-- 로트 분석 탭 네비게이션 -->
        <div class="lot-tab-navigation">
          <button 
            @click="lotAnalysisTab = 'low'" 
            :class="{ active: lotAnalysisTab === 'low' }"
            class="lot-tab-button"
          >
            불량률 낮은 로트 Top 5
          </button>
          <button 
            @click="lotAnalysisTab = 'high'" 
            :class="{ active: lotAnalysisTab === 'high' }"
            class="lot-tab-button"
          >
            불량률 높은 로트 Top 5
          </button>
        </div>
        
        <!-- 로트 분석 차트 -->
        <div class="lot-analysis-chart">
          <h5>{{ lotAnalysisTab === 'low' ? '불량률 낮은 로트 Top 5' : '불량률 높은 로트 Top 5' }}</h5>
          <div class="chart-legend">
            <div class="legend-item">
              <span class="legend-color" style="background: #4caf50;"></span>
              <span>정상</span>
            </div>
            <div class="legend-item">
              <span class="legend-color" style="background: #f44336;"></span>
              <span>Center</span>
            </div>
            <div class="legend-item">
              <span class="legend-color" style="background: #ff9800;"></span>
              <span>Donut</span>
            </div>
            <div class="legend-item">
              <span class="legend-color" style="background: #9c27b0;"></span>
              <span>Edge-Loc</span>
            </div>
            <div class="legend-item">
              <span class="legend-color" style="background: #2196f3;"></span>
              <span>Edge-Ring</span>
            </div>
            <div class="legend-item">
              <span class="legend-color" style="background: #607d8b;"></span>
              <span>Random</span>
            </div>
            <div class="legend-item">
              <span class="legend-color" style="background: #e91e63;"></span>
              <span>Scratch</span>
            </div>
            <div class="legend-item">
              <span class="legend-color" style="background: #ff5722;"></span>
              <span>Near-full</span>
            </div>
            <div class="legend-item">
              <span class="legend-color" style="background: #795548;"></span>
              <span>Loc</span>
            </div>
          </div>
          
          <div class="chart-container">
            <div v-for="(lot, index) in currentAnalysisLots" :key="lot.lotNumber" class="lot-analysis-row">
              <div class="lot-bar">
                <div class="lot-info">
                  <span class="lot-number">{{ lot.lotNumber }}</span>
                  <span class="defect-rate">{{ lot.defectRate }}%</span>
                </div>
                <div class="bar-container">
                  <div class="bar-background">
                    <div 
                      v-for="([defectType, count], index) in Object.entries(lot.defectTypes)" 
                      :key="defectType + index"
                      class="bar-fill defect-type"
                      :style="{ 
                        height: (count / lot.totalCount * 100) + '%',
                        bottom: getCumulativeHeight(index, lot.defectTypes, lot.totalCount),
                        backgroundColor: getDefectTypeColor(defectType)
                      }"
                      @mouseover="showTooltip($event, `${getDefectTypeName(defectType)}: ${count}개 (${(count / lot.totalCount * 100).toFixed(1)}%)`)"
                      @mousemove="moveTooltip($event)"
                      @mouseleave="hideTooltip"
                    ></div>
                  </div>

                  <div class="bar-labels">
                    <span class="normal-count">정상: {{ lot.normalCount }}</span>
                    <span class="defective-count">불량: {{ lot.defectiveCount }}</span>
                  </div>
                </div>
                <!-- Lot-level LLM actions -->
                <div
                  class="llm-lot-actions"
                >
                  <button
                    class="llm-button"
                    @click="requestLotExplanation(lot)"
                    :disabled="lotExplanationState(lot.lotNumber).loading"
                  >
                    <i class="fas fa-lightbulb"></i>
                    <span>{{ lotRequestLabel(lot.lotNumber) }}</span>
                  </button>
                  <p v-if="lotExplanationState(lot.lotNumber).error" class="llm-error">
                    {{ lotExplanationState(lot.lotNumber).error }}
                  </p>
                </div>
              </div>
              <div class="lot-defect-detail">
                <div class="lot-header">
                  <h6>{{ lot.lotNumber }} 인식 결과</h6>
                  <span class="total-defects">총 불량: {{ lot.defectiveCount }}개</span>
                </div>
                <div class="defect-types">
                  <div v-if="lot.defectiveCount === 0" class="no-defects">
                    불량이 없습니다.
                  </div>
                  <template v-else>
                    <template v-for="(count, type) in lot.defectTypes" :key="type">
                      <div v-if="count > 0" class="defect-type-item">
                        <span class="type-name">{{ getDefectTypeName(type) }}</span>
                        <span class="type-count">{{ count }}개</span>
                        <div class="type-bar">
                          <div 
                            class="type-fill" 
                            :style="{ 
                              width: (count / lot.defectiveCount * 100) + '%',
                              backgroundColor: getDefectTypeColor(type)
                            }"
                            :title="`${getDefectTypeName(type)}: ${count}개 (${(count / lot.defectiveCount * 100).toFixed(1)}%)`"
                          ></div>
                        </div>
                      </div>
                    </template>
                  </template>
                </div>
                <div v-if="lot.processHistory && lot.processHistory.length" class="lot-process-history">
                  <table class="process-history-table">
                    <thead>
                      <tr>
                        <th>Step</th>
                        <th>Tool</th>
                        <th>Recipe</th>
                      </tr>
                    </thead>
                    <tbody>
                      <tr v-for="(step, index) in lot.processHistory" :key="`${lot.lotNumber}-process-${index}`">
                        <td>{{ step.stepName }}</td>
                        <td>{{ step.tool }}</td>
                        <td>{{ step.recipe }}</td>
                      </tr>
                    </tbody>
                  </table>
                </div>
                <div v-else class="lot-process-history">
                  <p class="process-history-empty">No process history available.</p>
                </div>
                <!-- Lot-level LLM output -->
                <div
                  v-if="lotExplanationState(lot.lotNumber).text"
                  class="llm-output"
                >
                  <h6>LLM Suggestions</h6>
                  <div v-html="renderMarkdown(lotExplanationState(lot.lotNumber).text)"></div>
                </div>
              </div>
            </div>
          </div>
        </div>
        
        <!-- 불량 분석 도너트 차트 -->
        <div class="defect-analysis-section">
          <h5>불량 유형 분석</h5>
          <div class="donut-chart-container">
            <DonutChart :data="analysisChartData" />
            <div class="defect-summary">
              <div class="defect-rate-display">
                <div class="rate-value">{{ (analysisResult.defectiveRate * 100).toFixed(1) }}%</div>
                <div class="rate-label">전체 불량률</div>
              </div>
              <div class="defect-breakdown">
                <h6>불량 유형별 개수</h6>
                <div class="defect-list">
                  <div v-for="(count, type) in analysisResult.defectBreakdown" :key="type" class="defect-item">
                    <span class="defect-dot" :style="{ backgroundColor: getDefectColor(getDefectTypeName(type)) }"></span>
                    <span class="defect-name">{{ getDefectTypeName(type) }}</span>
                    <span class="defect-count">{{ count }}개</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
        
        <!-- 요약 결과 -->
        <div class="preview-summary">
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
          <div v-for="(item, index) in filteredHistory" :key="index" class="history-item" @click="toggleHistoryDetail(index)">
            <div class="history-info">
              <div class="history-meta">
                <span class="session">{{ item.sessionName }}</span>
                <span class="analysis-type">다중 로트</span>
                <i class="fas" :class="selectedHistoryIndex === index ? 'fa-chevron-up' : 'fa-chevron-down'" style="margin-left: auto;"></i>
              </div>
              <div class="history-details">
                <p>분석일시: {{ item.analysisDate }} | 파일 수: {{ item.fileCount }}개 | 로트 수: {{ item.lotCount }}개</p>
                <p>결과: 정상 {{ item.results.normal }}개, 불량 {{ item.results.defective }}개</p>
              </div>
            </div>
            
            <!-- 상세 내용 -->
            <div v-if="selectedHistoryIndex === index" class="history-detail-content">
              <div class="detail-section">
                <h6>분석 상세 정보</h6>
                <div class="detail-grid">
                  <div class="detail-item">
                    <span class="detail-label">로트 수:</span>
                    <span class="detail-value">{{ item.lotCount }}개</span>
                  </div>
                  <div class="detail-item">
                    <span class="detail-label">분석 일시:</span>
                    <span class="detail-value">{{ item.analysisDate }}</span>
                  </div>
                  <div class="detail-item">
                    <span class="detail-label">분석 파일 수:</span>
                    <span class="detail-value">{{ item.fileCount }}개</span>
                  </div>
                  <div class="detail-item">
                    <span class="detail-label">정상 결과:</span>
                    <span class="detail-value normal">{{ item.results.normal }}개</span>
                  </div>
                  <div class="detail-item">
                    <span class="detail-label">불량 결과:</span>
                    <span class="detail-value defective">{{ item.results.defective }}개</span>
                  </div>
                  <div class="detail-item">
                    <span class="detail-label">불량률:</span>
                    <span class="detail-value">{{ ((item.results.defective / item.fileCount) * 100).toFixed(1) }}%</span>
                  </div>
                  <div class="detail-item">
                    <span class="detail-label">평균 신뢰도:</span>
                    <span class="detail-value">{{ (85 + Math.random() * 15).toFixed(1) }}%</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
      </div>
      
      <!-- 이미지 모달 -->
      <div v-if="showImageModal" class="image-modal" @click="closeImageModal">
        <div class="modal-content" @click.stop>
          <button class="modal-close" @click="closeImageModal">
            <i class="fas fa-times"></i>
          </button>
          <img :src="modalImageUrl" alt="웨이퍼 이미지 원본" class="modal-image" />
        </div>
      </div>
    </div>
    
    <!-- 툴팁 -->
    <div v-if="tooltip.show" class="custom-tooltip" :style="{ left: tooltip.x + 'px', top: tooltip.y + 'px' }">
      {{ tooltip.text }}
    </div>
  </div>
</template>

<script setup>
import { ref, computed, reactive, onMounted, watch } from 'vue'
import DonutChart from '../components/DonutChart.vue'

const lotAnalysisTab = ref('low')
const selectedFile = ref(null)
const isAnalyzing = ref(false)
const selectedHistoryIndex = ref(null)
const showImageModal = ref(false)
const modalImageUrl = ref('')

const analysisResult = ref({
  isGenerated: false,
  totalCount: 0,
  normalCount: 0,
  defectiveCount: 0,
  defectiveRate: 0,
  defectBreakdown: {},
  lotData: [],
  topLots: [],
  worstLots: [],
  lotProcessHistory: {}
})

const analysisHistory = ref([
  {
    mode: 'multi-lot',
    sessionName: '2025-01-15 일괄분석',
    analysisDate: '2025-01-15 14:20:15',
    fileCount: 12,
    lotCount: 3,
    results: { normal: 9, defective: 3 }
  }
])

const GLOBAL_STATE_KEY = '__multiPredictionState__'


// Lot-level explanation state
const explanationStates = reactive({})

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

// Access or create per-lot explanation state
const lotExplanationState = (lotNumber) => {
  const key = lotNumber || 'unknown-lot'
  if (!Object.prototype.hasOwnProperty.call(explanationStates, key)) {
    explanationStates[key] = {
      text: '',
      error: '',
      loading: false
    }
  }
  return explanationStates[key]
}

// Reset explanations when data changes
const resetExplanationState = () => {
  Object.keys(explanationStates).forEach((key) => {
    delete explanationStates[key]
  })
}

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
    analysisHistory: JSON.parse(JSON.stringify(analysisHistory.value)),
    explanationStates: JSON.parse(JSON.stringify(explanationStates)),
    lotAnalysisTab: lotAnalysisTab.value,
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
  if (stored.analysisResult) {
    analysisResult.value = {
      ...analysisResult.value,
      ...stored.analysisResult,
      isGenerated: stored.analysisResult.isGenerated ?? true,
    }
  }
  if (Array.isArray(stored.analysisHistory)) {
    analysisHistory.value = stored.analysisHistory
  }
  lotAnalysisTab.value = stored.lotAnalysisTab || lotAnalysisTab.value
  resetExplanationState()
  Object.entries(stored.explanationStates || {}).forEach(([key, value]) => {
    explanationStates[key] = value
  })
}

onMounted(() => {
  restoreAnalysisState()
})

watch(lotAnalysisTab, () => {
  persistAnalysisState()
})

// Build payload for LLM explanation
const buildExplanationPayload = (lot) => {
  const defectCounts = {}

  Object.entries(lot.defectTypes || {}).forEach(([type, count]) => {
    const numericType = parseInt(type, 10)

    if (Number.isNaN(numericType)) {
      defectCounts[type] = count
      return
    }

    if (numericType === 8) {
      if (count > 0) {
        defectCounts.Normal = count
      }
      return
    }

    const label = getDefectTypeName(numericType)
    defectCounts[label] = count
  })

  const normalizedDefectRate =
    typeof lot.defectRate === 'number' && lot.defectRate > 1
      ? lot.defectRate / 100
      : lot.defectRate

  return {
    lotName: lot.lotNumber,
    defectCounts,
    normalCount: lot.normalCount,
    defectiveCount: lot.defectiveCount,
    defectRate: typeof normalizedDefectRate === 'number' ? normalizedDefectRate : 0,
  }
}

// Request LLM explanation for a lot
const requestLotExplanation = async (lot) => {
  if (!lot || typeof lot.defectRate !== 'number') {
    return
  }

  const state = lotExplanationState(lot.lotNumber)
  state.error = ''
  state.text = ''
  state.loading = true

  try {
    const payload = buildExplanationPayload(lot)
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
    state.text = data.explanation || ''
  } catch (error) {
    console.error('LLM request failed:', error)
    state.error = error.message || 'Failed to retrieve suggestions.'
  } finally {
    state.loading = false
    persistAnalysisState()
  }
}

// Label helper for request button
const lotRequestLabel = (lotNumber) =>
  lotExplanationState(lotNumber).loading ? 'Requesting...' : 'Request LLM Suggestions'

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

const getCumulativeHeight = (index, defectTypes, totalCount) => {
  const entries = Object.entries(defectTypes)

  let cumulative = 0
  for (let i = 0; i < index; i++) {
    const count = entries[i][1]
    cumulative += (count / totalCount) * 100
  }
  return `${cumulative}%`
}

const showHistorySearch = ref(false)
const searchFilters = ref({
  analysisDate: ''
})

const tooltip = ref({
  show: false,
  x: 0,
  y: 0,
  text: ''
})

const showTooltip = (event, text) => {
  tooltip.value.show = true
  tooltip.value.text = text
  tooltip.value.x = event.clientX + 10
  tooltip.value.y = event.clientY - 10
}

const moveTooltip = (event) => {
  tooltip.value.x = event.clientX + 10
  tooltip.value.y = event.clientY - 10
}

const hideTooltip = () => {
  tooltip.value.show = false
}

const analysisChartData = computed(() => {
  if (!analysisResult.value.isGenerated) return { num_by_cat: {} }
  
  const num_by_cat = {}
  
  // 정상 데이터 추가
  if (analysisResult.value.normalCount > 0) {
    num_by_cat['none'] = analysisResult.value.normalCount
  }
  
  // 불량 데이터 추가
  Object.entries(analysisResult.value.defectBreakdown).forEach(([type, count]) => {
    if (count > 0) {
      num_by_cat[getDefectTypeName(type)] = count
    }
  })
  
  return { num_by_cat }
})

const getDefectColor = (type) => {
  const defectTypes = ['정상', 'Center', 'Donut', 'Edge-Loc', 'Edge-Ring', 'Loc', 'Random', 'Scratch', 'Near-full']
  const colors = ['#4caf50', '#f44336', '#ff9800', '#9c27b0', '#2196f3', '#795548', '#607d8b', '#e91e63', '#ff5722']
  const index = defectTypes.indexOf(type)
  return colors[index] || colors[1]
}

const getDefectTypeColor = (type) => {
  const defectTypeColors = {
    0: '#f44336',    // Center
    1: '#ff9800',    // Donut
    2: '#9c27b0',    // Edge-Loc
    3: '#2196f3',    // Edge-Ring
    4: '#795548',    // Loc
    5: '#607d8b',    // Random
    6: '#e91e63',    // Scratch
    7: '#ff5722',    // Near-full
    8: '#4caf50',     // 정상
    'Center': '#f44336',
    'Donut': '#ff9800', 
    'Edge-Loc': '#9c27b0',
    'Edge-Ring': '#2196f3',
    'Loc': '#795548',
    'Random': '#607d8b',
    'Scratch': '#e91e63',
    'Near-full': '#ff5722',
    '정상': '#4caf50'
  }
  return defectTypeColors[type] || '#ffffff'
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
    7: 'Near-full',
    8: '정상'
  }
  return defectTypeNames[type] || 'Unknown'
}

const availableDates = computed(() => {
  const dates = analysisHistory.value.map(item => item.analysisDate.split(' ')[0])
  return [...new Set(dates)].sort((a, b) => new Date(b) - new Date(a))
})

const hasActiveFilters = computed(() => {
  return searchFilters.value.analysisDate
})

const filteredHistory = computed(() => {
  let filtered = analysisHistory.value
  
  if (searchFilters.value.analysisDate) {
    filtered = filtered.filter(item => {
      const itemDate = item.analysisDate.split(' ')[0]
      return itemDate === searchFilters.value.analysisDate
    })
  }
  
  return filtered
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
  persistAnalysisState()
}

const toggleHistorySearch = () => {
  showHistorySearch.value = !showHistorySearch.value
}

const applyFilters = () => {
  // 필터링은 computed에서 자동으로 처리됨
}

const resetFilters = () => {
  searchFilters.value = {
    analysisDate: ''
  }
}

const toggleHistoryDetail = (index) => {
  selectedHistoryIndex.value = selectedHistoryIndex.value === index ? null : index
}

const closeImageModal = () => {
  showImageModal.value = false
  modalImageUrl.value = ''
}

const currentAnalysisLots = computed(() => {
  return lotAnalysisTab.value === 'low' ? analysisResult.value.topLots : analysisResult.value.worstLots;
});

const analyzeImages = async () => {
  if (!selectedFile.value) {
    return;
  }

  // Clear previous explanation state for a fresh analysis
  resetExplanationState()
  persistAnalysisState()
  isAnalyzing.value = true;
  
  const formData = new FormData();
  formData.append('file', selectedFile.value);

  try {
    const response = await fetch('http://127.0.0.1:8001/predict_multi_images_multi_lots', {
      method: 'POST',
      body: formData,
    });

    if (response.ok) {
      const data = await response.json();
      const lotProcessHistoryMap = data.lotProcessHistory || {};
      
      const totalCount = data.fileCount;
      const normalCount = data.normal;
      const defectiveCount = data.defective;
      const defectiveRate = data.defectRate;
      
      // 불량 유형별 개수 계산
      const defectBreakdown = {};
      data.predictions.forEach(pred => {
        if (pred >= 0 && pred <= 7) {
          defectBreakdown[pred] = (defectBreakdown[pred] || 0) + 1;
        }
      });
      
      // ✅ lot_defect_ranking 순서 유지하여 lotData 생성 (이미 불량률 높은 순)
      const lotData = [];

      Object.entries(data.lot_defect_ranking).forEach(([lotNumber, defectCounts]) => {
        // 0~7번 오류의 총합만 defectiveCount로 계산
        const defectiveCount = Object.entries(defectCounts)
          .filter(([key]) => parseInt(key) >= 0 && parseInt(key) <= 7)
          .reduce((sum, [, count]) => sum + count, 0);

        // 정상 개수 (8번): defectCounts에 존재하면 그대로, 없으면 0
        const normalCount = defectCounts[8] || 0;

        const lotTotalCount = defectiveCount + normalCount;

        const defectRate = parseFloat(((defectiveCount / lotTotalCount) * 100).toFixed(1));
        const normalRate = parseFloat(((normalCount / lotTotalCount) * 100).toFixed(1));

        lotData.push({
          lotNumber,
          totalCount: lotTotalCount,
          normalCount,
          defectiveCount,
          defectRate,
          normalRate,
          defectTypes: defectCounts,  // 0~8 모두 포함
          processHistory: normalizeProcessHistory(lotProcessHistoryMap[lotNumber])
        });
      });

      // ✅ Top 5 (불량률 낮은 순)
      const topLots = [...lotData]
        .sort((a, b) => a.defectRate - b.defectRate)
        .slice(0, 5);

      // ✅ Worst 5 (불량률 높은 순)
      const worstLots = lotData.slice(0, 5); // 이미 정렬된 상태

      // ✅ 결과 저장
      analysisResult.value = {
        isGenerated: true,
        totalCount,
        normalCount,
        defectiveCount,
        defectiveRate,
        defectBreakdown,
        lotData,
        topLots,
        worstLots,
        lotProcessHistory: lotProcessHistoryMap
      };
      
      const lotCount = Object.keys(data.lot_defect_ranking || {}).length;
      const historyItem = {
        mode: 'multi-lot',
        sessionName: `자동분석_${new Date().toLocaleDateString('ko-KR')}`,
        analyst: '시스템',
        purpose: '자동 일괄분석',
        analysisDate: new Date().toLocaleString('ko-KR'),
        fileCount: totalCount,
        lotCount: lotCount,
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
</script>

<style scoped>
.multi-prediction {
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

.result-section {
  background: #f8f9fa;
  padding: 2rem;
  border-radius: 10px;
  margin: 2rem 0;
}

.lot-tab-navigation {
  display: flex;
  justify-content: flex-end;
  margin: 1rem 0;
  border-bottom: 2px solid #e9ecef;
}

.lot-tab-button {
  background: none;
  border: none;
  padding: 0.75rem 1.5rem;
  cursor: pointer;
  font-size: 0.9rem;
  font-weight: 500;
  color: #666;
  border-bottom: 3px solid transparent;
  transition: all 0.3s ease;
}

.lot-tab-button:hover {
  color: #007bff;
  background: #f8f9fa;
}

.lot-tab-button.active {
  color: #007bff;
  border-bottom-color: #007bff;
  font-weight: 600;
}

.lot-analysis-chart {
  margin-bottom: 2rem;
}

.chart-legend {
  display: flex;
  flex-wrap: wrap;
  gap: 1rem;
  margin-bottom: 1rem;
  justify-content: center;
}

.legend-item {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  font-size: 0.9rem;
}

.legend-color {
  width: 16px;
  height: 16px;
  border-radius: 3px;
  display: inline-block;
}

.chart-container {
  margin-top: 1rem;
  display: flex;
  flex-direction: column;
  gap: 2rem;
}

.lot-analysis-row {
  display: flex;
  gap: 2rem;
  align-items: flex-start;
  padding: 1rem;
  background: white;
  border-radius: 10px;
  border: 1px solid #e9ecef;
}

.lot-bar {
  flex-shrink: 0;
  width: 180px;
  display: flex;
  flex-direction: column;
  align-items: center;
  min-height: 350px;
}

.lot-defect-detail {
  flex: 1;
  padding: 1rem;
  background: #f8f9fa;
  border-radius: 8px;
  border: 1px solid #e9ecef;
}

/* Process history table styling */
.lot-process-history {
  margin-top: 1.5rem;
}

.process-history-table {
  width: 100%;
  border-collapse: collapse;
  background: #ffffff;
  border-radius: 6px;
  overflow: hidden;
  border: 1px solid #dee2e6;
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

.lot-info {
  text-align: center;
  margin-bottom: 1rem;
  width: 100%;
}

.lot-number {
  font-weight: 600;
  color: #333;
  display: block;
  margin-bottom: 0.5rem;
}

.defect-rate {
  font-weight: 600;
  color: #d32f2f;
  font-size: 1.1rem;
}

.bar-container {
  flex: 1;
  display: flex;
  flex-direction: column;
  justify-content: flex-end;
  align-items: center;
  width: 100%;
  position: relative;
}

.bar-background {
  width: 50px;
  height: 250px;
  background: #f5f5f5;
  border-radius: 25px;
  overflow: hidden;
  position: relative;
  display: flex;
  flex-direction: column-reverse;
}

.bar-fill {
  width: 100%;
  position: absolute;
}

.bar-fill.normal {
  z-index: 0;
  background: linear-gradient(180deg, #4caf50, #66bb6a);
}

.bar-fill.defect-type {
  z-index: 1;
  transition: height 0.3s ease;
}

.bar-labels {
  display: flex;
  flex-direction: column;
  align-items: center;
  margin-top: 0.5rem;
  font-size: 0.8rem;
  gap: 0.25rem;
}

.normal-count {
  color: #4caf50;
  font-weight: 500;
}

.defective-count {
  color: #f44336;
  font-weight: 500;
}

.lot-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 1rem;
  padding-bottom: 0.5rem;
  border-bottom: 1px solid #ddd;
}

.lot-header h6 {
  margin: 0;
  color: #333;
  font-size: 1.1rem;
}

.total-defects {
  color: #f44336;
  font-weight: 600;
  font-size: 0.9rem;
}

.defect-types {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
}

.defect-type-item {
  display: flex;
  align-items: center;
  margin-bottom: 0.75rem;
}

.type-name {
  flex: 1;
  font-size: 0.9rem;
  color: #333;
}

.type-count {
  margin-right: 1rem;
  font-weight: 600;
  color: #666;
  min-width: 40px;
  text-align: right;
}

.type-bar {
  flex: 2;
  height: 12px;
  background: #f5f5f5;
  border-radius: 6px;
  overflow: hidden;
  min-width: 100px;
}

.type-fill {
  height: 100%;
  min-width: 2px;
  transition: width 0.3s ease;
  border-radius: 6px;
}

.no-defects {
  text-align: center;
  color: #666;
  font-style: italic;
  padding: 1rem;
  background: #f8f9fa;
  border-radius: 6px;
  border: 1px solid #e9ecef;
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

.filter-group input,
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

.history-count {
  margin-bottom: 1rem;
  font-weight: 600;
  color: #666;
}

.history-item {
  background: white;
  border: 1px solid #ddd;
  border-radius: 10px;
  padding: 1.5rem;
  margin-bottom: 1rem;
  cursor: pointer;
  transition: all 0.3s ease;
}

.history-item:hover {
  border-color: #007bff;
  box-shadow: 0 2px 8px rgba(0,123,255,0.1);
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

.factory {
  background: #e3f2fd;
  color: #1976d2;
}

.type {
  background: #f3e5f5;
  color: #7b1fa2;
}

.analysis-type {
  background: #fff3e0;
  color: #f57c00;
}

.session {
  background: #e8f5e8;
  color: #388e3c;
}

.history-details p {
  margin: 0.5rem 0;
  color: #666;
}

.history-detail-content {
  margin-top: 1.5rem;
  padding-top: 1.5rem;
  border-top: 1px solid #e9ecef;
}

.detail-section h6 {
  margin-bottom: 1rem;
  color: #333;
  font-size: 1.1rem;
}

.detail-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
  gap: 1rem;
}

.detail-item {
  display: flex;
  justify-content: space-between;
  padding: 0.75rem;
  background: #f8f9fa;
  border-radius: 6px;
}

.detail-label {
  font-weight: 600;
  color: #666;
}

.detail-value {
  color: #333;
}

.detail-value.normal {
  color: #28a745;
  font-weight: 600;
}

.detail-value.defective {
  color: #dc3545;
  font-weight: 600;
}

.image-modal {
  position: fixed;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  background: rgba(0, 0, 0, 0.8);
  display: flex;
  justify-content: center;
  align-items: center;
  z-index: 1000;
}

.modal-content {
  position: relative;
  max-width: 90%;
  max-height: 90%;
}

.modal-close {
  position: absolute;
  top: -40px;
  right: 0;
  background: white;
  border: none;
  border-radius: 50%;
  width: 40px;
  height: 40px;
  cursor: pointer;
  font-size: 1.2rem;
  color: #333;
}

.modal-image {
  max-width: 100%;
  max-height: 100%;
  border-radius: 10px;
}

.custom-tooltip {
  position: fixed;
  background: rgba(0, 0, 0, 0.8);
  color: white;
  padding: 8px 12px;
  border-radius: 4px;
  font-size: 12px;
  pointer-events: none;
  z-index: 1000;
  white-space: nowrap;
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
  margin-top: 1.5rem;
}

.llm-output h6 {
  margin-bottom: 0.5rem;
  font-weight: 700;
}

.llm-lot-actions {
  margin-top: 1rem;
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 0.5rem;
  width: 100%;
}

@media (max-width: 768px) {
  .filter-grid {
    grid-template-columns: 1fr;
  }
  
  .history-meta {
    flex-wrap: wrap;
  }
  
  .history-header {
    flex-direction: column;
    gap: 1rem;
    align-items: stretch;
  }
  
  .summary-cards {
    grid-template-columns: repeat(2, 1fr);
  }
  
  .lot-bar {
    min-height: 180px;
  }
  
  .bar-background {
    height: 100px;
  }
  
  .donut-chart-container {
    flex-direction: column;
    align-items: center;
  }
  
  .defect-list {
    grid-template-columns: 1fr;
  }
}
</style>