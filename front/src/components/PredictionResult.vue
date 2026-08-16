// src/components/CNNPrediction/PredictionResult.vue
<template>
  <BCard title="분류 결과" class="h-100">
    <div v-if="imagePreviewUrl" class="text-center mb-3">
      <img :src="imagePreviewUrl" alt="업로드된 이미지" class="img-fluid" style="max-height: 300px;">
    </div>
    
    <div v-if="predictionResult && predictionResult.isDetected" class="mt-3">
      <h5>분류 결과:</h5>
      <p><strong>상태:</strong> {{ predictionResult.isDefective ? '불량' : '정상' }}</p>
      <p><strong>불량 유형:</strong> {{ predictionResult.defectType || '없음' }}</p>
      <p><strong>신뢰도:</strong> {{ predictionResult.confidence || 0 }}%</p>
    </div>
    
    <div v-else-if="!imagePreviewUrl" class="text-center text-muted">
      <p>이미지를 업로드하고 불량 분류를 시작하여 결과가 표시됩니다.</p>
    </div>
  </BCard>
</template>

<script setup>
import { BCard } from 'bootstrap-vue-next';

// Props 정의
defineProps({
  imagePreviewUrl: {
    type: [String, null],
    default: null
  },
  predictionResult: {
    type: [Object, null],
    default: null
  }
});
</script>

<style scoped>
.img-fluid {
  border-radius: 8px;
  box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}
</style>