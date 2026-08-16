import React, { useEffect, useRef } from 'react';
import { motion } from 'framer-motion';
import {
  Chart as ChartJS,
  ArcElement,
  Tooltip,
  CategoryScale,
  LinearScale,
} from 'chart.js';
import { Pie } from 'react-chartjs-2';

ChartJS.register(ArcElement, Tooltip, CategoryScale, LinearScale);

interface PredictionPieChartProps {
  probabilities: {
    Nondemented: number;
    Converted: number;
    Demented: number;
  };
}

export const PredictionPieChart: React.FC<PredictionPieChartProps> = ({ probabilities }) => {
  const chartRef = useRef<ChartJS<'pie'>>(null);

  const data = {
    labels: ['Nondemented', 'Converted', 'Demented'],
    datasets: [
      {
        label: 'Probability',
        data: [
          probabilities.Nondemented * 100,
          probabilities.Converted * 100,
          probabilities.Demented * 100,
        ],
        backgroundColor: [
          'rgba(16, 185, 129, 0.8)',
          'rgba(251, 191, 36, 0.8)',
          'rgba(244, 63, 94, 0.8)',
        ],
        borderColor: [
          'rgba(16, 185, 129, 1)',
          'rgba(251, 191, 36, 1)',
          'rgba(244, 63, 94, 1)',
        ],
        borderWidth: 2,
        hoverBackgroundColor: [
          'rgba(16, 185, 129, 0.95)',
          'rgba(251, 191, 36, 0.95)',
          'rgba(244, 63, 94, 0.95)',
        ],
      },
    ],
  };

  const options = {
    responsive: false,
    maintainAspectRatio: false,
    animation: {
      animateRotate: true,
      animateScale: true,
      duration: 1500,
      easing: 'easeInOutQuart' as const,
    },
    plugins: {
      legend: { display: false }, // legend disabled (handled manually in ResultCard)
      tooltip: {
        backgroundColor: 'rgba(0, 0, 0, 0.8)',
        titleColor: 'rgba(255, 255, 255, 0.95)',
        bodyColor: 'rgba(255, 255, 255, 0.8)',
        borderColor: 'rgba(255, 255, 255, 0.2)',
        borderWidth: 1,
        padding: 12,
        cornerRadius: 12,
        callbacks: {
          label: (context: any) => {
            const label = context.label || '';
            const value = context.parsed || 0;
            return `${label}: ${value.toFixed(1)}%`;
          },
        },
      },
    },
    elements: {
      arc: { borderRadius: 8 },
    },
  };

  useEffect(() => {
    chartRef.current?.update();
  }, [probabilities]);

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.9 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ delay: 0.4, duration: 0.5 }}
      className="relative w-[220px] h-[220px]"
    >
      <Pie ref={chartRef} data={data} options={options} />
    </motion.div>
  );
};
