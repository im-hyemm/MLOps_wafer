// src/App.vue
<template>
  <div id="app">
    <!-- 홈 페이지 내용 -->
    <div v-if="isHomePage" class="home-container">
      <div class="main-content">
        <!-- 제목 섹션 -->
        <div class="title-section">
          <h1 class="main-title">WaferGuard</h1>
          <p class="subtitle">AIOps 기반 반도체 공정 개선 서비스</p>
        </div>

        <!-- 서비스 카드 섹션 -->
        <div class="service-cards">
          <div class="card" @click="navigateTo('/single-prediction')">
            <div class="card-icon">
              <i class="fas fa-image"></i>
            </div>
            <h3><b>Image</b></h3>
            <p>하나의 웨이퍼 이미지 <br></br>불량 여부 및 불량 유형 분석</p>
          </div>

          <div class="card" @click="navigateTo('/multi-single-lot')">
            <div class="card-icon">
              <i class="fas fa-images"></i>
            </div>
            <h3><b>Lot</b></h3>
            <p>하나의 로트에서 나온 <br></br>여러 웨이퍼 이미지를 일괄 분석</p>
          </div>

          <div class="card" @click="navigateTo('/multi-multi-lot')">
            <div class="card-icon">
              <i class="fas fa-layer-group"></i>
            </div>
            <h3><b>Factory</b></h3>
            <p>여러 로트에서 나온 <br></br>웨이퍼 이미지들 일괄 분석</p>
          </div>

          <div class="card" @click="navigateTo('/PKLPrediction')">
            <div class="card-icon">
              <i class="fas fa-database"></i>
            </div>
            <h3><b>Model Management</b></h3>
            <p>라벨링된 데이터를 통해 <br></br>모델 재학습 필요 여부를 판단</p>
          </div>
        </div>

        <!-- 모델 재학습 추천 섹션 -->
        <div class="retrain-section">
          <div class="retrain-card">
            <div class="retrain-content">
              <h3>모델 재학습 기능</h3>
              <p>최신 데이터로 모델 성능을 향상시키세요. 새로운 웨이퍼 타입이나 변화된 패턴을 반영하여 더 정확한 분석 결과를 얻을 수 있습니다.</p>
            </div>
          </div>
        </div>

        <!-- 기능 소개 섹션 -->
        <div class="features-section">
          <div class="features-grid">
            <div class="feature-item">
              <i class="fas fa-brain"></i>
              <h4>AI 기반 분석</h4>
              <p>딥러닝 모델을 활용한 웨이퍼 타입 분류</p>
            </div>
            <div class="feature-item">
              <i class="fas fa-chart-bar"></i>
              <h4>상세한 결과</h4>
              <p>분석 결과와 함께 신뢰도 및 공정 정보 제공</p>
            </div>
            <div class="feature-item">
              <i class="fas fa-bolt"></i>
              <h4>LLM 기반 불량 원인 진단</h4>
              <p>최신 논문을 기반으로 불량 원인을 빠르게 분석</p>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- 다른 페이지용 탭 네비게이션 -->
    <div v-if="!isHomePage" class="tab-navigation">
      <div class="tab-container">
        <router-link to="/" class="tab-item" :class="{ active: route.path === '/' }">
          Home
        </router-link>
        <router-link to="/single-prediction" class="tab-item" :class="{ active: route.path === '/single-prediction' }">
          Image
        </router-link>
        <router-link to="/multi-single-lot" class="tab-item" :class="{ active: route.path === '/multi-single-lot' }">
          Lot
        </router-link>
        <router-link to="/multi-multi-lot" class="tab-item" :class="{ active: route.path === '/multi-multi-lot' }">
          Factory
        </router-link>
        <router-link to="/PKLPrediction" class="tab-item" :class="{ active: route.path === '/PKLPrediction' }">
          Model Management
        </router-link>
      </div>
    </div>

    <router-view v-if="!isHomePage"></router-view>
  </div>
</template>




<script setup>
import { computed } from 'vue'
import { useRoute, useRouter } from 'vue-router'

const route = useRoute()
const router = useRouter()
const isHomePage = computed(() => route.path === '/')

const navigateTo = (path) => {
  router.push(path)
}

const startRetraining = () => {
  alert('모델 재학습을 시작합니다.')
}
</script>

<style scoped>
#app {
  font-family: Avenir, Helvetica, Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
}

.tab-navigation {
  background: #f8f9fa;
  border-bottom: 1px solid #dee2e6;
  padding: 0;
}

.tab-container {
  display: flex;
  max-width: 1200px;
  margin: 0 auto;
}

.tab-item {
  flex: 1;
  padding: 1rem 2rem;
  text-decoration: none;
  color: #6c757d;
  background: #f8f9fa;
  border: 1px solid #dee2e6;
  border-bottom: none;
  text-align: center;
  font-weight: bold;
  font-size: 1.1rem;
  transition: all 0.3s ease;
  position: relative;
}

.tab-item:hover {
  background: #e9ecef;
  color: #495057;
}

.tab-item.active {
  background: white;
  color: #007bff;
  font-weight: bold;
  font-size: 1.1rem;
  border-bottom: 2px solid #007bff;
}

.tab-item:not(:last-child) {
  border-right: none;
}

@media (max-width: 768px) {
  .tab-item {
    padding: 0.75rem 1rem;
    font-size: 0.9rem;
  }
}

/* 홈 페이지 스타일 */
.home-container {
  min-height: 100vh;
  background: white;
  padding: 2rem 0;
}

.main-content {
  max-width: 1200px;
  margin: 0 auto;
  padding: 0 2rem;
}

.title-section {
  text-align: center;
  margin-bottom: 4rem;
  color: #333;
}

.main-title {
  font-size: 3rem;
  font-weight: bold;
  margin-bottom: 1rem;
  text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
}

.subtitle {
  font-size: 1.2rem;
  opacity: 0.9;
  margin-bottom: 0;
}

.service-cards {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 1.5rem;
  margin-bottom: 4rem;
}

.card {
  background: white;
  border-radius: 15px;
  padding: 1.5rem;
  text-align: center;
  box-shadow: 0 10px 30px rgba(0,0,0,0.2);
  transition: transform 0.3s ease, box-shadow 0.3s ease;
  cursor: pointer;
  min-height: 200px;
  display: flex;
  flex-direction: column;
  justify-content: center;
}

.card:hover {
  transform: translateY(-10px);
  box-shadow: 0 20px 40px rgba(0,0,0,0.3);
}

.card-icon {
  font-size: 2.5rem;
  color: #667eea;
  margin-bottom: 0.8rem;
}

.card h3 {
  font-size: 1.2rem;
  margin-bottom: 0.8rem;
  color: #333;
  line-height: 1.3;
}

.card p {
  color: #666;
  line-height: 1.4;
  font-size: 0.9rem;
  margin: 0;
}

.features-section {
  text-align: center;
  color: #333;
}

.features-section h2 {
  font-size: 2.5rem;
  margin-bottom: 3rem;
  text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
}

.features-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
  gap: 2rem;
}

.feature-item {
  background: #f8f9fa;
  border-radius: 15px;
  padding: 2rem;
  border: 1px solid #e9ecef;
  box-shadow: 0 2px 10px rgba(0,0,0,0.1);
}

.feature-item i {
  font-size: 2.5rem;
  margin-bottom: 1rem;
  color: #667eea;
}

.feature-item h4 {
  font-size: 1.3rem;
  margin-bottom: 1rem;
}

.feature-item p {
  color: #666;
  line-height: 1.6;
}

.retrain-section {
  margin-bottom: 4rem;
}

.retrain-card {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  border-radius: 15px;
  padding: 2rem;
  color: white;
  display: flex;
  justify-content: space-between;
  align-items: center;
  box-shadow: 0 10px 30px rgba(0,0,0,0.2);
}

.retrain-content h3 {
  font-size: 1.5rem;
  margin-bottom: 0.5rem;
}

.retrain-content p {
  opacity: 0.9;
  line-height: 1.6;
  margin: 0;
}

.retrain-btn {
  background: white;
  color: #667eea;
  border: none;
  border-radius: 10px;
  padding: 1rem 2rem;
  font-size: 1rem;
  font-weight: bold;
  cursor: pointer;
  transition: transform 0.3s ease, box-shadow 0.3s ease;
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.retrain-btn:hover {
  transform: translateY(-2px);
  box-shadow: 0 5px 15px rgba(0,0,0,0.2);
}

@media (max-width: 1200px) {
  .service-cards {
    grid-template-columns: repeat(2, 1fr);
  }
}

@media (max-width: 768px) {
  .main-title {
    font-size: 2rem;
  }
  
  .service-cards {
    grid-template-columns: 1fr;
  }
  
  .features-grid {
    grid-template-columns: 1fr;
  }
  
  .retrain-card {
    flex-direction: column;
    text-align: center;
    gap: 1.5rem;
  }
}
</style>