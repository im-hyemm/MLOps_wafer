<template>
  <div class="model-status-section">
    <h3>모델 상태</h3>
    <div class="status-card">
      <div class="status-info">
        <div class="info-item">
          <span class="info-label">현재 버전:</span>
          <span class="version-badge">{{ status.version }}</span>
        </div>
        <div class="info-item">
          <span class="info-label">최종 학습일:</span>
          <span class="info-value">{{ status.lastTrained }}</span>
        </div>
      </div>
      
      <div class="update-history">
        <h5>업데이트 이력</h5>
        <div class="history-table">
          <div class="table-header">
            <div class="header-cell">날짜</div>
            <div class="header-cell">설명</div>
          </div>
          <div class="table-body">
            <div v-for="log in status.updateLogs" :key="log.date" class="table-row">
              <div class="table-cell">{{ log.date }}</div>
              <div class="table-cell">{{ log.description }}</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { defineProps } from 'vue';

const props = defineProps({
  status: Object,
});
</script>

<style scoped>
.model-status-section {
  margin-top: 2rem;
}

.status-card {
  background: white;
  border: 1px solid #e9ecef;
  border-radius: 10px;
  padding: 2rem;
  box-shadow: 0 2px 10px rgba(0,0,0,0.1);
}

.status-info {
  margin-bottom: 2rem;
}

.info-item {
  display: flex;
  align-items: center;
  margin-bottom: 1rem;
  gap: 1rem;
}

.info-label {
  font-weight: 600;
  color: #333;
  min-width: 120px;
}

.version-badge {
  background: #007bff;
  color: white;
  padding: 0.25rem 0.75rem;
  border-radius: 15px;
  font-size: 0.9rem;
  font-weight: 500;
}

.info-value {
  color: #666;
}

.update-history h5 {
  margin-bottom: 1rem;
  color: #333;
  border-bottom: 1px solid #e9ecef;
  padding-bottom: 0.5rem;
}

.history-table {
  border: 1px solid #e9ecef;
  border-radius: 8px;
  overflow: hidden;
}

.table-header {
  display: grid;
  grid-template-columns: 150px 1fr;
  background: #f8f9fa;
  border-bottom: 1px solid #e9ecef;
}

.header-cell {
  padding: 0.75rem 1rem;
  font-weight: 600;
  color: #333;
  border-right: 1px solid #e9ecef;
}

.header-cell:last-child {
  border-right: none;
}

.table-body {
  background: white;
}

.table-row {
  display: grid;
  grid-template-columns: 150px 1fr;
  border-bottom: 1px solid #e9ecef;
}

.table-row:last-child {
  border-bottom: none;
}

.table-cell {
  padding: 0.75rem 1rem;
  color: #666;
  border-right: 1px solid #e9ecef;
}

.table-cell:last-child {
  border-right: none;
}

@media (max-width: 768px) {
  .table-header,
  .table-row {
    grid-template-columns: 1fr;
  }
  
  .header-cell,
  .table-cell {
    border-right: none;
    border-bottom: 1px solid #e9ecef;
  }
  
  .info-item {
    flex-direction: column;
    align-items: flex-start;
    gap: 0.5rem;
  }
  
  .info-label {
    min-width: auto;
  }
}
</style>