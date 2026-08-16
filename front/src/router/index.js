import { createRouter, createWebHistory } from 'vue-router';
import HomePage from '../App.vue';
import SinglePrediction from '../pages/SinglePrediction.vue';
import MultiSingleLot from '../pages/MultiSingleLot.vue';
import MultiPrediction from '../pages/MultiPrediction.vue';
import PKLPrediction from '../pages/PKLPrediction.vue';

const routes = [
  {
    path: '/',
    name: 'HomePage',
    component: HomePage
  },
  {
    path: '/HomePage',
    redirect: '/'
  },
  {
    path: '/single-prediction',
    name: 'SinglePrediction',
    component: SinglePrediction
  },
  {
    path: '/multi-single-lot',
    name: 'MultiSingleLot',
    component: MultiSingleLot
  },
  {
    path: '/multi-multi-lot',
    name: 'MultiMultiLot',
    component: MultiPrediction
  },
  {
    path: '/multi-prediction',
    redirect: '/multi-single-lot'
  },
  {
    path: '/PKLPrediction',
    name: 'PKLPrediction',
    component: PKLPrediction
  }
];

const router = createRouter({
  history: createWebHistory(),
  routes
});

export default router;