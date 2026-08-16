<script setup>
import { onMounted, watch, ref } from 'vue'
import { Chart, ArcElement, Tooltip, Legend, DoughnutController, CategoryScale, Title} from 'chart.js'
import ChartDataLabels from 'chartjs-plugin-datalabels';

Chart.register(ArcElement, Tooltip, Legend, DoughnutController, CategoryScale, Title, ChartDataLabels)

const chartRef = ref(null)
let donutChart = null
const DEFECT_TYPE_COLORS = {
  'Center': '#f44336',
  'Donut': '#ff9800',
  'Edge-Loc': '#9c27b0',
  'Edge-Ring': '#2196f3',
  'Loc': '#795548',
  'Random': '#607d8b',
  'Scratch': '#e91e63',
  'Near-full': '#ff5722',
  'none': '#4caf50',
}

const props = defineProps({
  data: {
    type: Object,
    required: true
  }
})

const getChartData = () => {
  const numByCat = props.data.num_by_cat || {}
  const labels = Object.keys(props.data.num_by_cat)
  const values = Object.values(props.data.num_by_cat)
  const backgroundColors = labels.map(type => DEFECT_TYPE_COLORS[type] || '#ccc')

  return {
    labels,
    datasets: [
      {
        data: values,
        backgroundColor: backgroundColors,
        borderWidth: 1
      }
    ]
  }
}

const renderChart = () => {
  const ctx = chartRef.value.getContext('2d')
  if (donutChart) donutChart.destroy()

  donutChart = new Chart(ctx, {
    type: 'doughnut',
    data: getChartData(),
    options: {
      responsive: true,
      plugins: {
        tooltip: {
          callbacks: {
            label: function (context) {
              const label = context.label || ''
              const value = context.raw
              const total = context.dataset.data.reduce((a, b) => a + b, 0)
              const percentage = ((value / total) * 100).toFixed(1)
              return `${label}: ${value}개 (${percentage}%)`
            }
          }
        },
        legend: {
          display: false,
          position: 'right'
        },
        datalabels: {
          display: true,  // 라벨 표시 여부
          color: '#000',  // 라벨 색상
          formatter: (value, context) => {
            const data = context.chart.data.datasets[0].data;
            const total = data.reduce((acc, val) => acc + val, 0);
            const percentage = ((value / total) * 100).toFixed(1); // 소수점 1자리
            return `${percentage}%`;
          },
          font: {
            // weight: 'bold',
            size: 14,
          },
        },
      },
      cutout: '60%'  // 내부 빈 공간 비율
    }
  })
}

onMounted(() => {
  if (props.data && props.data.num_by_cat) {
    renderChart()
  }
})

watch(() => props.data, () => {
  if (props.data && props.data.num_by_cat) {
    renderChart()
  }
})
</script>

<template>
  <div class="donut-chart-wrapper">
    <canvas ref="chartRef" width="300" height="300"></canvas>
  </div>
</template>

<style scoped>
.donut-chart-wrapper {
  position: relative;
  width: 300px;
  height: 300px;
}
</style>
