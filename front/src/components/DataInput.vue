// src/components/CNNPrediction/DataInput.vue
<template>
  <BCard title="웨이퍼 이미지 업로드" class="h-100">
    <BFormInput 
      v-model="file" 
      type="file"
      placeholder="이미지 파일을 선택하세요..."
      accept="image/*"
      @change="handleFileUpload"
    ></BFormInput>
    
    <div class="mt-3">
      <p class="text-muted">분석할 웨이퍼 이미지를 업로드하세요.</p>
    </div>
    
    <BButton 
      variant="primary" 
      class="w-100 mt-3" 
      @click="$emit('predict-start')" 
      :disabled="isLoading"
    >
      <BSpinner small v-if="isLoading" class="me-2"></BSpinner>
      {{ isLoading ? '분류 중...' : '불량 분류 시작' }}
    </BButton>
  </BCard>
</template>

<script setup>
import { ref, defineProps, defineEmits } from 'vue';
import { BCard, BFormInput, BButton, BSpinner } from 'bootstrap-vue-next';

// 부모 컴포넌트로부터 받는 Props
const props = defineProps({
  isLoading: Boolean,
});

// 부모 컴포넌트로 전달하는 이벤트
const emit = defineEmits(['file-uploaded', 'predict-start']);

const file = ref(null);

const handleFileUpload = (event) => {
  const file = event.target.files[0];
  if (file) {
    emit('file-uploaded', file);
  }
};
</script>