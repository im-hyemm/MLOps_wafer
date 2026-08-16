<template>
  <div class="single-prediction">
    <div class="container">
      <h1>단일 이미지 분석</h1>
      <p>단일 웨이퍼 이미지를 업로드하여 불량 여부와 유형을 확인하세요.</p>
      
      <!-- 분석 섹션 -->
      <div class="tab-content">

      <!-- 이미지 업로드 -->
      <div class="upload-area">
        <input 
          type="file" 
          accept="image/*" 
          @change="handleFileUpload"
          class="file-input"
          id="file-input"
        >
        <label for="file-input" class="upload-label">
          <i class="fas fa-image"></i>
          <span>웨이퍼 이미지를 선택하세요.</span>
        </label>
        <div class="multi-lot-notice">
          <i class="fas fa-info-circle"></i>
          <span>이미지 파일명에 로트 정보와 웨이퍼 ID가 포함되어야 합니다. (예: lot100_15.png)</span>
        </div>
      </div>

      <!-- 선택된 파일 정보 -->
      <div v-if="selectedFile" class="file-info">
        <h3>선택된 파일</h3>
        <div class="file-details">
          <img v-if="imagePreviewUrl" :src="imagePreviewUrl" alt="미리보기" class="preview-image" />
          <div class="file-meta">
            <p><strong>파일명:</strong> {{ selectedFile.name }}</p>
            <p><strong>크기:</strong> {{ formatFileSize(selectedFile.size) }}</p>
            <button @click="removeImage" class="remove-btn">이미지 삭제</button>
          </div>
        </div>

        <div class="action-buttons">
          <button @click="predictImage" :disabled="!isFormValid" class="analyze-btn">
            <i class="fas fa-play"></i> 분석 시작
          </button>
        </div>
      </div>

      <!-- 분석 중 -->
      <div v-if="isAnalyzing" class="loading">
        <i class="fas fa-spinner fa-spin"></i>
        <span>이미지 분석 중...</span>
      </div>

      <!-- 분석 결과 -->
      <div v-if="previewResult.isGenerated" class="result-section">
        <h3>분석 결과</h3>
        <div class="preview-result-content">
          <!-- 분석된 이미지 -->
          <div v-if="previewResult.coloredImg" class="preview-image-section">
            <h5>분석 결과 이미지</h5>
            <!-- ⭐Modal trigger disabled -->
            <img 
              :src="previewResult.coloredImg" 
              alt="분석된 웨이퍼 이미지" 
              class="preview-thumbnail"
            />
            <!-- ⭐Modal hint removed
            <p class="image-click-hint">이미지를 클릭하면 원본 크기로 볼 수 있습니다</p>
            -->
          </div>
          
          <!-- 분석 결과 -->
          <div class="preview-card">
            <div class="preview-status" :class="previewResult.isDefective ? 'defective' : 'normal'">
              {{ previewResult.isDefective ? '불량' : '정상' }}
            </div>
            <div class="preview-details">
              <p><strong>로트 번호:</strong> {{ previewResult.lotNumber }}</p>
              <p><strong>웨이퍼 ID:</strong> {{ previewResult.waferId }}</p>
              <p><strong>불량 유형:</strong> {{ previewResult.defectType }}</p>
              <p><strong>신뢰도:</strong> {{ previewResult.confidence }}%</p>
            </div>
          </div>
        </div>

        <!-- Process history -->
        <div class="process-history-section">
          <h5>공정 이력</h5>
          <table v-if="previewResult.processHistory && previewResult.processHistory.length" class="process-history-table">
            <thead>
              <tr>
                <th>Step</th>
                <th>Tool</th>
                <th>Recipe</th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="(step, index) in previewResult.processHistory" :key="`single-preview-process-${index}`">
                <td>{{ step.stepName }}</td>
                <td>{{ step.tool }}</td>
                <td>{{ step.recipe }}</td>
              </tr>
            </tbody>
          </table>
          <p v-else class="process-history-empty">No process history available.</p>
        </div>
      </div>



<!-- 과거 이력 -->


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
              <label>웨이퍼 ID</label>
              <input v-model="searchFilters.waferId" type="text" placeholder="웨이퍼 ID 입력" />
            </div>
            <div class="filter-group">
              <label>로트 번호</label>
              <div class="autocomplete-container">
                <input 
                  v-model="searchFilters.lotNumber" 
                  type="text" 
                  placeholder="로트 번호 입력/선택"
                  @input="filterLotSuggestions"
                  @focus="showAllLots"
                  @blur="hideLotSuggestions"
                />
                <div v-if="showLotSuggestions" class="suggestions-dropdown">
                  <div v-if="filteredLots.length === 0" class="no-suggestions">
                    일치하는 로트 번호가 없습니다
                  </div>
                  <div 
                    v-for="lot in filteredLots" 
                    :key="lot"
                    class="suggestion-item"
                    @click="selectLot(lot)"
                  >
                    {{ lot }}
                  </div>
                </div>
              </div>
            </div>
            <div class="filter-group">
              <label>분석일시</label>
              <input 
                v-model="searchFilters.analysisDate" 
                type="date" 
                class="date-picker"
              />
            </div>
            <div class="filter-group">
              <label>결과</label>
              <select v-model="searchFilters.result">
                <option value="">전체</option>
                <option value="정상">정상</option>
                <option value="불량">불량</option>
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
                <div class="meta-tags">
                  <span class="lot-number">{{ item.lotNumber }}</span>
                  <span class="wafer-id">{{ item.waferId }}</span>
                  <span class="analysis-type">단일 이미지</span>
                </div>
                <i class="fas" :class="selectedHistoryIndex === index ? 'fa-chevron-up' : 'fa-chevron-down'"></i>
              </div>
              <div class="history-details">
                <div class="detail-row">
                  <span class="detail-item">웨이퍼 ID: {{ item.waferId }}</span>
                  <span class="detail-separator">|</span>
                  <span class="detail-item">로트: {{ item.lotNumber }}</span>
                </div>
                <div class="detail-row">
                  <span class="detail-item">분석일시: {{ item.analysisDate }}</span>
                </div>
                <div class="detail-row">
                  <span class="detail-item">불량 유형: <span :class="item.result.isDefective ? 'defective' : 'normal'">{{ item.result.defectType }}</span></span>
                  <span class="detail-separator">|</span>
                  <span class="detail-item">신뢰도: {{ item.result.confidence }}%</span>
                </div>
              </div>
            </div>
            
            <!-- 상세 내용 -->
            <div v-if="selectedHistoryIndex === index" class="history-detail-content">
              <div class="detail-section">
                <h6>분석 상세 정보</h6>
                <div class="detail-grid">
                  <div class="detail-item">
                    <span class="detail-label">웨이퍼 ID:</span>
                    <span class="detail-value">{{ item.waferId }}</span>
                  </div>
                  <div class="detail-item">
                    <span class="detail-label">로트 번호:</span>
                    <span class="detail-value">{{ item.lotNumber }}</span>
                  </div>
                  <div class="detail-item">
                    <span class="detail-label">분석 일시:</span>
                    <span class="detail-value">{{ item.analysisDate }}</span>
                  </div>
                  <div class="detail-item">
                    <span class="detail-label">불량 유형:</span>
                    <span class="detail-value" :class="item.result.isDefective ? 'defective' : 'normal'">{{ item.result.defectType }}</span>
                  </div>
                  <div class="detail-item">
                    <span class="detail-label">신뢰도:</span>
                    <span class="detail-value">{{ item.result.confidence }}%</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
      </div>
      
      <!-- 이미지 모달 -->
            <!-- ⭐Modal disabled
      <div v-if="showImageModal" class="image-modal" @click="closeImageModal">
        <div class="modal-content" @click.stop>
          <button class="modal-close" @click="closeImageModal">
            <i class="fas fa-times"></i>
          </button>
          <img :src="previewResult.coloredImg || imagePreviewUrl" alt="웨이퍼 이미지" class="modal-image" />
        </div>
      </div>
      -->
    </div>
  </div>
</template>

<script setup>
import { ref, computed } from 'vue'

const activeTab = ref('analysis')
const selectedFile = ref(null)
const imagePreviewUrl = ref(null)
const isAnalyzing = ref(false)

const waferInfo = ref({
  factory: '',
  type: ''
})

const predictionResult = ref({
  filename: '',
  prediction: '',
  confidence: 0,
})

const analysisTime = ref(0)

const showHistorySearch = ref(false)
const searchFilters = ref({
  waferId: '',
  lotNumber: '',
  analysisDate: '',
  result: ''
})

const showFactorySuggestions = ref(false)
const showLotSuggestions = ref(false)
const showTypeSuggestions = ref(false)
const filteredFactories = ref([])
const filteredLots = ref([])
const filteredTypes = ref([])

const showPreviewFactorySuggestions = ref(false)
const showPreviewLotSuggestions = ref(false)
const showPreviewWaferSuggestions = ref(false)
const showPreviewTypeSuggestions = ref(false)
const filteredPreviewFactories = ref([])
const filteredPreviewLots = ref([])
const filteredPreviewWafers = ref([])
const filteredPreviewTypes = ref([])

const previewInfo = ref({
  factory: '',
  type: '',
  lotNumber: '',
  waferId: ''
})

const previewImageUrl = ref(null)
// ⭐Modal disabled
// const showImageModal = ref(false)
const selectedHistoryIndex = ref(null)

const previewResult = ref({
  isGenerated: false,
  isDefective: false,
  defectType: '정상',
  confidence: 0,
  estimatedTime: 0,
  defectiveRate: 0,
  defectBreakdown: {},
  processHistory: []
})

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

const extractFileInfo = (filename) => {
  const nameWithoutExt = filename.split('.')[0];
  const parts = nameWithoutExt.split('_');
  
  if (parts.length >= 2) {
    return {
      lotNumber: parts[0],
      waferId: parts[1]
    };
  }
  
  return {
    lotNumber: 'UNKNOWN',
    waferId: nameWithoutExt
  };
};

const predictImage = async () => {
  if (!selectedFile.value) {
    return;
  }

  isAnalyzing.value = true;
  previewResult.value.isGenerated = false;
  
  const startTime = performance.now();
  const formData = new FormData();
  formData.append('file', selectedFile.value);

  try {
    const response = await fetch('http://127.0.0.1:8001/predict_img', {
      method: 'POST',
      body: formData,
    });

    if (response.ok) {
      const data = await response.json();
      const lotProcessHistory = normalizeProcessHistory(data.lotProcessHistory);
      const endTime = performance.now();
      analysisTime.value = ((endTime - startTime) / 1000).toFixed(2);
      
      const fileInfo = extractFileInfo(selectedFile.value.name);
      
      // previewResult에 결과 저장
      previewResult.value = {
        isGenerated: true,
        isDefective: data.prediction !== 'none',
        defectType: data.prediction,
        confidence: (data.confidence * 100).toFixed(1),
        estimatedTime: analysisTime.value,
        lotNumber: fileInfo.lotNumber,
        waferId: fileInfo.waferId,
        coloredImg: data.colored_img,
        processHistory: lotProcessHistory
      };
      
      // 이력에 추가
      analysisHistory.value.unshift({
        waferId: fileInfo.waferId,
        lotNumber: fileInfo.lotNumber,
        analysisDate: new Date().toLocaleString('ko-KR'),
        result: { 
          isDefective: data.prediction !== 'none', 
          defectType: data.prediction,
          confidence: (data.confidence * 100).toFixed(1)
        }
      });
    } else {
      console.error('예측 요청 실패:', response.statusText);
    }
  } catch (error) {
    console.error('예측 요청 실패:', error);
  } finally {
    isAnalyzing.value = false;
  }
};


const analysisHistory = ref([
  {
    waferId: 'W001',
    lotNumber: 'lot4521',
    analysisDate: '2025-01-15 14:30:25',
    result: { isDefective: false, defectType: 'none', confidence: '95.2' }
  },
  {
    waferId: 'W002',
    lotNumber: 'lot6549',
    analysisDate: '2025-01-15 13:15:10',
    result: { isDefective: true, defectType: 'Scratch', confidence: '87.8' }
  },
  {
    waferId: 'W003',
    lotNumber: 'lot7890',
    analysisDate: '2025-01-14 16:45:30',
    result: { isDefective: false, defectType: 'none', confidence: '92.1' }
  },
  {
    waferId: 'W004',
    lotNumber: 'lot4675',
    analysisDate: '2025-01-14 11:20:15',
    result: { isDefective: true, defectType: 'Edge-Loc', confidence: '89.5' }
  }
])

const isFormValid = computed(() => {
  return selectedFile.value
})

const isPreviewFormValid = computed(() => {
  return previewImageUrl.value
})

const hasActiveFilters = computed(() => {
  return searchFilters.value.waferId || 
         searchFilters.value.lotNumber || 
         searchFilters.value.analysisDate || 
         searchFilters.value.result
})

const filteredHistory = computed(() => {
  let filtered = analysisHistory.value
  
  if (searchFilters.value.waferId) {
    filtered = filtered.filter(item => 
      item.waferId.toLowerCase().includes(searchFilters.value.waferId.toLowerCase())
    )
  }
  
  if (searchFilters.value.lotNumber) {
    filtered = filtered.filter(item => 
      item.lotNumber.toLowerCase().includes(searchFilters.value.lotNumber.toLowerCase())
    )
  }
  
  if (searchFilters.value.analysisDate) {
    filtered = filtered.filter(item => {
      const itemDate = item.analysisDate.split(' ')[0]
      return itemDate === searchFilters.value.analysisDate
    })
  }
  
  if (searchFilters.value.result) {
    filtered = filtered.filter(item => {
      if (searchFilters.value.result === '정상') {
        return !item.result.isDefective
      } else if (searchFilters.value.result === '불량') {
        return item.result.isDefective
      }
      return true
    })
  }
  
  return filtered
})

const handleFileUpload = (event) => {
  selectedFile.value = event.target.files[0]
  if (selectedFile.value) {
    const reader = new FileReader()
    reader.onload = (e) => {
      imagePreviewUrl.value = e.target.result
    }
    reader.readAsDataURL(selectedFile.value)
  }
}

const removeImage = () => {
  selectedFile.value = null
  imagePreviewUrl.value = null
  previewResult.value.isGenerated = false
  analysisTime.value = 0
  document.getElementById('file-input').value = ''
}

const analyzeImage = () => {
  isAnalyzing.value = true
  
  setTimeout(() => {
    const isDefective = Math.random() > 0.5
    const defectTypes = ['정상', 'Center', 'Donut', 'Edge-Loc', 'Edge-Ring', 'Loc', 'Random', 'Scratch', 'Near-full']
    
    const result = {
      isDetected: true,
      isDefective: isDefective,
      defectType: isDefective ? defectTypes[Math.floor(Math.random() * (defectTypes.length - 1)) + 1] : '정상',
      confidence: (85 + Math.random() * 15).toFixed(2),
    }
    
    predictionResult.value = result
    
    // 이력에 추가
    analysisHistory.value.unshift({
      factory: waferInfo.value.factory,
      waferType: waferInfo.value.type,
      waferId: waferInfo.value.waferId,
      lotNumber: waferInfo.value.lotNumber,
      analysisDate: new Date().toLocaleString('ko-KR'),
      result: { isDefective: result.isDefective, defectType: result.defectType }
    })
    
    isAnalyzing.value = false
  }, 3000)
}

const toggleHistorySearch = () => {
  showHistorySearch.value = !showHistorySearch.value
}

const applyFilters = () => {
  // 필터링은 computed에서 자동으로 처리됨
}

const resetFilters = () => {
  searchFilters.value = {
    waferId: '',
    lotNumber: '',
    analysisDate: '',
    result: ''
  }
}

const uniqueFactories = computed(() => {
  return [...new Set(analysisHistory.value.map(item => item.factory))]
})

const uniqueLots = computed(() => {
  return [...new Set(analysisHistory.value.map(item => item.lotNumber))]
})

const uniqueWafers = computed(() => {
  return [...new Set(analysisHistory.value.map(item => item.waferId))]
})

const uniqueTypes = computed(() => {
  return [...new Set(analysisHistory.value.map(item => item.waferType))]
})



const filterFactorySuggestions = () => {
  const query = searchFilters.value.factory.toLowerCase()
  if (query === '') {
    filteredFactories.value = uniqueFactories.value
  } else {
    filteredFactories.value = uniqueFactories.value.filter(factory => 
      factory.toLowerCase().includes(query)
    )
  }
}

const filterLotSuggestions = () => {
  const query = searchFilters.value.lotNumber.toLowerCase()
  if (query === '') {
    filteredLots.value = uniqueLots.value
  } else {
    filteredLots.value = uniqueLots.value.filter(lot => 
      lot.toLowerCase().includes(query)
    )
  }
}

const selectFactory = (factory) => {
  searchFilters.value.factory = factory
  showFactorySuggestions.value = false
}

const selectLot = (lot) => {
  searchFilters.value.lotNumber = lot
  showLotSuggestions.value = false
}



const hideFactorySuggestions = () => {
  setTimeout(() => {
    showFactorySuggestions.value = false
  }, 150)
}

const hideLotSuggestions = () => {
  setTimeout(() => {
    showLotSuggestions.value = false
  }, 150)
}



const showAllFactories = () => {
  filteredFactories.value = uniqueFactories.value
  showFactorySuggestions.value = true
}

const showAllLots = () => {
  filteredLots.value = uniqueLots.value
  showLotSuggestions.value = true
}

const filterTypeSuggestions = () => {
  const query = waferInfo.value.type.toLowerCase()
  if (query === '') {
    filteredTypes.value = uniqueTypes.value
  } else {
    filteredTypes.value = uniqueTypes.value.filter(type => 
      type.toLowerCase().includes(query)
    )
  }
}

const showAllTypes = () => {
  filteredTypes.value = uniqueTypes.value
  showTypeSuggestions.value = true
}

const selectType = (type) => {
  waferInfo.value.type = type
  showTypeSuggestions.value = false
}

const hideTypeSuggestions = () => {
  setTimeout(() => {
    showTypeSuggestions.value = false
  }, 150)
}

const filterPreviewFactorySuggestions = () => {
  const query = previewInfo.value.factory.toLowerCase()
  if (query === '') {
    filteredPreviewFactories.value = uniqueFactories.value
  } else {
    filteredPreviewFactories.value = uniqueFactories.value.filter(factory => 
      factory.toLowerCase().includes(query)
    )
  }
}

const filterPreviewLotSuggestions = () => {
  const query = previewInfo.value.lotNumber.toLowerCase()
  if (query === '') {
    filteredPreviewLots.value = uniqueLots.value
  } else {
    filteredPreviewLots.value = uniqueLots.value.filter(lot => 
      lot.toLowerCase().includes(query)
    )
  }
}

const filterPreviewWaferSuggestions = () => {
  const query = previewInfo.value.waferId.toLowerCase()
  if (query === '') {
    filteredPreviewWafers.value = uniqueWafers.value
  } else {
    filteredPreviewWafers.value = uniqueWafers.value.filter(wafer => 
      wafer.toLowerCase().includes(query)
    )
  }
}

const showAllPreviewFactories = () => {
  filteredPreviewFactories.value = uniqueFactories.value
  showPreviewFactorySuggestions.value = true
}

const showAllPreviewLots = () => {
  filteredPreviewLots.value = uniqueLots.value
  showPreviewLotSuggestions.value = true
}

const showAllPreviewWafers = () => {
  filteredPreviewWafers.value = uniqueWafers.value
  showPreviewWaferSuggestions.value = true
}

const selectPreviewFactory = (factory) => {
  previewInfo.value.factory = factory
  showPreviewFactorySuggestions.value = false
}

const selectPreviewLot = (lot) => {
  previewInfo.value.lotNumber = lot
  showPreviewLotSuggestions.value = false
}

const selectPreviewWafer = (wafer) => {
  previewInfo.value.waferId = wafer
  showPreviewWaferSuggestions.value = false
}

const hidePreviewFactorySuggestions = () => {
  setTimeout(() => {
    showPreviewFactorySuggestions.value = false
  }, 150)
}

const hidePreviewLotSuggestions = () => {
  setTimeout(() => {
    showPreviewLotSuggestions.value = false
  }, 150)
}

const hidePreviewWaferSuggestions = () => {
  setTimeout(() => {
    showPreviewWaferSuggestions.value = false
  }, 150)
}

const filterPreviewTypeSuggestions = () => {
  const query = previewInfo.value.type.toLowerCase()
  if (query === '') {
    filteredPreviewTypes.value = uniqueTypes.value
  } else {
    filteredPreviewTypes.value = uniqueTypes.value.filter(type => 
      type.toLowerCase().includes(query)
    )
  }
}

const showAllPreviewTypes = () => {
  filteredPreviewTypes.value = uniqueTypes.value
  showPreviewTypeSuggestions.value = true
}

const selectPreviewType = (type) => {
  previewInfo.value.type = type
  showPreviewTypeSuggestions.value = false
}

const hidePreviewTypeSuggestions = () => {
  setTimeout(() => {
    showPreviewTypeSuggestions.value = false
  }, 150)
}



const handlePreviewImageUpload = (event) => {
  const file = event.target.files[0]
  if (file) {
    const reader = new FileReader()
    reader.onload = (e) => {
      previewImageUrl.value = e.target.result
    }
    reader.readAsDataURL(file)
  }
}

// ⭐Modal handlers disabled
// const openImageModal = () => {
//   showImageModal.value = true
//   // 모달에서 실제 업로드된 이미지를 보여주기 위해 previewImageUrl 대신 imagePreviewUrl 사용
// }
// 
// const closeImageModal = () => {
//   showImageModal.value = false
// }

const donutSegments = computed(() => {
  if (!previewResult.value.isGenerated) return []
  
  const defectTypes = ['정상', '스크래치', '오염', '파티클', '크랙', '버블', '얼룩', '열손상', '전기적결함']
  const colors = ['#4caf50', '#f44336', '#ff9800', '#9c27b0', '#2196f3', '#795548', '#607d8b', '#e91e63', '#ff5722']
  
  const total = 100 // 단일 이미지이므로 100%로 계산
  let currentAngle = 0
  const segments = []
  
  // 정상 세그먼트
  const normalPercentage = (100 - previewResult.value.defectiveRate).toFixed(1)
  const normalAngle = ((100 - previewResult.value.defectiveRate) / 100) * 2 * Math.PI
  
  if (normalPercentage > 0) {
    segments.push({
      type: '정상',
      percentage: normalPercentage,
      path: createArcPath(currentAngle, currentAngle + normalAngle, 80, 120),
      color: colors[0],
      labelX: Math.cos(currentAngle + normalAngle / 2) * 100,
      labelY: Math.sin(currentAngle + normalAngle / 2) * 100
    })
    currentAngle += normalAngle
  }
  
  // 불량 세그먼트들
  Object.entries(previewResult.value.defectBreakdown).forEach(([type, count], index) => {
    if (count > 0) {
      const percentage = previewResult.value.defectiveRate.toFixed(1)
      const angle = (previewResult.value.defectiveRate / 100) * 2 * Math.PI
      const colorIndex = defectTypes.indexOf(type)
      
      segments.push({
        type,
        percentage,
        path: createArcPath(currentAngle, currentAngle + angle, 80, 120),
        color: colors[colorIndex] || colors[1],
        labelX: Math.cos(currentAngle + angle / 2) * 100,
        labelY: Math.sin(currentAngle + angle / 2) * 100
      })
      currentAngle += angle
    }
  })
  
  return segments
})

const createArcPath = (startAngle, endAngle, innerRadius, outerRadius) => {
  const x1 = Math.cos(startAngle) * outerRadius
  const y1 = Math.sin(startAngle) * outerRadius
  const x2 = Math.cos(endAngle) * outerRadius
  const y2 = Math.sin(endAngle) * outerRadius
  const x3 = Math.cos(endAngle) * innerRadius
  const y3 = Math.sin(endAngle) * innerRadius
  const x4 = Math.cos(startAngle) * innerRadius
  const y4 = Math.sin(startAngle) * innerRadius
  
  const largeArc = endAngle - startAngle > Math.PI ? 1 : 0
  
  return `M ${x1} ${y1} A ${outerRadius} ${outerRadius} 0 ${largeArc} 1 ${x2} ${y2} L ${x3} ${y3} A ${innerRadius} ${innerRadius} 0 ${largeArc} 0 ${x4} ${y4} Z`
}

const getDefectColor = (type) => {
  const defectTypes = ['정상', 'Center', 'Donut', 'Edge-Loc', 'Edge-Ring', 'Loc', 'Random', 'Scratch', 'Near-full']
  const colors = ['#4caf50', '#f44336', '#ff9800', '#9c27b0', '#2196f3', '#795548', '#607d8b', '#e91e63', '#ff5722']
  const index = defectTypes.indexOf(type)
  return colors[index] || colors[1]
}

const toggleHistoryDetail = (index) => {
  selectedHistoryIndex.value = selectedHistoryIndex.value === index ? null : index
}

const generatePreview = () => {
  const isDefective = Math.random() > 0.6
  const defectTypes = ['Center', 'Donut', 'Edge-Loc', 'Edge-Ring', 'Loc', 'Random', 'Scratch', 'Near-full']
  
  let defectiveRate = 0
  let defectBreakdown = {}
  
  if (isDefective) {
    defectiveRate = (Math.random() * 30 + 5).toFixed(1) // 5-35% 불량률
    const selectedDefectType = defectTypes[Math.floor(Math.random() * defectTypes.length)]
    defectBreakdown[selectedDefectType] = 1
  }
  
  previewResult.value = {
    isGenerated: true,
    isDefective: isDefective,
    defectType: isDefective ? Object.keys(defectBreakdown)[0] : '정상',
    confidence: (80 + Math.random() * 20).toFixed(1),
    estimatedTime: Math.floor(2 + Math.random() * 4),
    defectiveRate: parseFloat(defectiveRate),
    defectBreakdown
  }
}

const formatFileSize = (bytes) => {
  if (bytes === 0) return '0 Bytes'
  const k = 1024
  const sizes = ['Bytes', 'KB', 'MB', 'GB']
  const i = Math.floor(Math.log(bytes) / Math.log(k))
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i]
}
</script>

<style scoped>
.single-prediction {
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
  color: #007bff;
  background: #f8f9fa;
}

.tab-button.active {
  color: #007bff;
  border-bottom-color: #007bff;
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

.preview-btn {
  background: #28a745;
  color: white;
  border: none;
  padding: 1rem 2rem;
  border-radius: 5px;
  cursor: pointer;
  font-size: 1rem;
  margin-top: 1rem;
}

.preview-btn:disabled {
  background: #ccc;
  cursor: not-allowed;
}

.preview-result {
  background: white;
  padding: 2rem;
  border-radius: 10px;
  margin-top: 2rem;
  border: 1px solid #e9ecef;
}

.preview-card {
  display: flex;
  align-items: center;
  gap: 2rem;
  margin: 1rem 0;
}

.preview-status {
  padding: 1rem 2rem;
  border-radius: 10px;
  font-size: 1.3rem;
  font-weight: bold;
}

.preview-status.normal {
  background: #d4edda;
  color: #155724;
}

.preview-status.defective {
  background: #f8d7da;
  color: #721c24;
}

.preview-details p {
  margin: 0.5rem 0;
  color: #333;
}

.preview-upload {
  margin: 1.5rem 0;
  border: 2px dashed #ddd;
  border-radius: 8px;
  padding: 1.5rem;
  text-align: center;
}

.preview-upload-label {
  cursor: pointer;
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 0.5rem;
  color: #666;
}

.preview-upload-label i {
  font-size: 2rem;
  color: #28a745;
}

.preview-result-content {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 2rem;
  margin: 1rem 0;
}

.preview-image-section {
  text-align: center;
}

.preview-image-section h5 {
  margin-bottom: 1rem;
  color: #333;
}

.preview-thumbnail {
  width: 100%;
  max-width: 260px;
  max-height: 260px;
  height: auto;
  object-fit: contain;
  border-radius: 10px;
  border: 2px solid #e9ecef;
  cursor: default;
  background-color: #ffffff;
}

.multi-lot-notice i {
  margin-right: 6px; /* 아이콘과 글자 사이 간격 */
}

/* ⭐Hover interaction removed */
.image-click-hint {
  font-size: 0.8rem;
  color: #666;
  margin-top: 0.5rem;
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
  max-width: 80vw;
  max-height: 80vh;
  padding: 1rem;
  background: #ffffff;
  border-radius: 12px;
  box-shadow: 0 10px 30px rgba(0, 0, 0, 0.25);
  display: flex;
  justify-content: center;
  align-items: center;
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
  width: auto;
  height: auto;
  max-width: 70vw;
  max-height: 70vh;
  border-radius: 10px;
  object-fit: contain;
}

.defect-analysis-section {
  margin-top: 2rem;
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
  grid-template-columns: 1fr;
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

.preview-note {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  margin-top: 1.5rem;
  padding: 1rem;
  background: #fff3cd;
  border: 1px solid #ffeaa7;
  border-radius: 5px;
  color: #856404;
  font-size: 0.9rem;
}

@media (max-width: 768px) {
  .preview-result-content {
    grid-template-columns: 1fr;
  }
  
  .preview-thumbnail {
    width: 150px;
    height: 150px;
  }
  
  .donut-chart-container {
    flex-direction: column;
    align-items: center;
  }
}

.meta-info-section {
  background: #f8f9fa;
  padding: 2rem;
  border-radius: 10px;
  margin-bottom: 2rem;
}

.form-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 1rem;
  margin-top: 1rem;
}

.form-group label {
  display: block;
  margin-bottom: 0.5rem;
  font-weight: 600;
  color: #333;
}

.form-group input,
.form-group select {
  width: 100%;
  padding: 0.75rem;
  border: 1px solid #ddd;
  border-radius: 5px;
  font-size: 1rem;
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

.file-info {
  margin: 2rem 0;
}

.file-details {
  display: flex;
  gap: 2rem;
  align-items: center;
  margin: 1rem 0;
}

.preview-image {
  width: 150px;
  height: 150px;
  object-fit: cover;
  border-radius: 10px;
  border: 1px solid #ddd;
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

.action-buttons {
  display: flex;
  justify-content: center;
  margin-top: 1rem;
}

.remove-btn {
  background: #dc3545;
  color: white;
  border: none;
  padding: 0.5rem 1rem;
  border-radius: 5px;
  cursor: pointer;
  font-size: 0.9rem;
  margin-top: 0.5rem;
}

.remove-btn:hover {
  background: #c82333;
}

.loading {
  text-align: center;
  font-size: 1.2rem;
  color: #007bff;
  margin: 2rem 0;
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

.result-card {
  display: flex;
  align-items: center;
  gap: 2rem;
}

.result-status {
  padding: 1rem 2rem;
  border-radius: 10px;
  font-size: 1.5rem;
  font-weight: bold;
}

.result-status.normal {
  background: #d4edda;
  color: #155724;
}

.result-status.defective {
  background: #f8d7da;
  color: #721c24;
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

.history-count {
  margin-bottom: 1rem;
  font-weight: 600;
  color: #666;
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
  cursor: pointer;
  transition: all 0.3s ease;
}

.history-item:hover {
  border-color: #007bff;
  box-shadow: 0 2px 8px rgba(0,123,255,0.1);
}

.history-meta {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 1rem;
}

.meta-tags {
  display: flex;
  gap: 0.75rem;
  align-items: center;
  flex-wrap: wrap;
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
  grid-template-columns: repeat(3, 1fr);
  grid-template-rows: auto auto;
  gap: 1rem;
}

.detail-grid .detail-item:nth-child(4),
.detail-grid .detail-item:nth-child(5) {
  grid-column: span 1;
}

.detail-grid .detail-item:nth-child(4) {
  grid-column: 1 / 2;
}

.detail-grid .detail-item:nth-child(5) {
  grid-column: 2 / 3;
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

.history-meta span {
  padding: 0.25rem 0.75rem;
  border-radius: 15px;
  font-size: 0.9rem;
  font-weight: 500;
}

.lot-number {
  background: #e3f2fd;
  color: #1976d2;
}

.wafer-id {
  background: #f3e5f5;
  color: #7b1fa2;
}

.analysis-type {
  background: #e8f5e8;
  color: #388e3c;
}

.history-details {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
}

.detail-row {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  color: #666;
  font-size: 0.95rem;
}

.detail-item {
  display: inline-flex;
  align-items: center;
}

.detail-separator {
  color: #ccc;
  margin: 0 0.25rem;
}

.normal {
  color: #28a745;
  font-weight: 600;
}

.defective {
  color: #dc3545;
  font-weight: 600;
}

.autocomplete-container {
  position: relative;
}

.suggestions-dropdown {
  position: absolute;
  top: 100%;
  left: 0;
  right: 0;
  background: white;
  border: 1px solid #ddd;
  border-top: none;
  border-radius: 0 0 5px 5px;
  max-height: 200px;
  overflow-y: auto;
  z-index: 1000;
  box-shadow: 0 2px 8px rgba(0,0,0,0.1);
}

.suggestion-item {
  padding: 0.75rem 1rem;
  cursor: pointer;
  border-bottom: 1px solid #f0f0f0;
  transition: background-color 0.2s ease;
}

.suggestion-item:hover {
  background-color: #f8f9fa;
}

.suggestion-item:last-child {
  border-bottom: none;
}

.no-suggestions {
  padding: 0.75rem 1rem;
  color: #999;
  font-style: italic;
  text-align: center;
}

.date-picker {
  width: 100%;
  padding: 0.5rem;
  border: 1px solid #ddd;
  border-radius: 5px;
  font-size: 0.9rem;
}

@media (max-width: 768px) {
  .form-grid {
    grid-template-columns: 1fr;
  }
  
  .file-details {
    flex-direction: column;
    text-align: center;
  }
  
  .result-card {
    flex-direction: column;
    text-align: center;
  }
  
  .history-meta {
    flex-direction: column;
    align-items: flex-start;
    gap: 0.75rem;
  }
  
  .meta-tags {
    width: 100%;
  }
  
  .detail-row {
    flex-wrap: wrap;
    gap: 0.25rem;
  }
}
/* Process history table styling */
.process-history-section {
  margin-top: 2rem;
  border-top: 1px solid #e9ecef;
  padding-top: 1.5rem;
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