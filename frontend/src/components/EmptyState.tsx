import React from 'react';
import { motion } from 'framer-motion';

interface EmptyStateProps {
  onAnalyze: () => void;
  loading: boolean;
}

export const EmptyState: React.FC<EmptyStateProps> = ({ onAnalyze, loading }) => {
  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.96 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.5, ease: [0.16, 1, 0.3, 1] }}
      className="relative"
    >
      <div className="relative bg-black/40 backdrop-blur-2xl rounded-[2rem] shadow-[0_8px_32px_rgba(0,0,0,0.4)] border border-white/10 p-10 md:p-12 h-full flex items-center justify-center min-h-[600px] overflow-hidden">
        {/* Gradient overlay layers */}
        <div className="absolute inset-0 bg-gradient-to-br from-white/5 via-transparent to-black/20 pointer-events-none" />
        <div className="absolute inset-0 bg-gradient-to-t from-blue-500/5 via-transparent to-transparent pointer-events-none" />
        
        {/* Subtle inner glow */}
        <div className="absolute inset-[1px] rounded-[calc(2rem-1px)] bg-gradient-to-b from-white/10 to-transparent pointer-events-none" />
        
        <div className="text-center relative z-10">
          <motion.div
            animate={{ 
              scale: [1, 1.06, 1],
              opacity: [0.5, 0.8, 0.5]
            }}
            transition={{ 
              duration: 3.5, 
              repeat: Infinity,
              ease: 'easeInOut'
            }}
            className="mx-auto w-24 h-24 mb-10 text-white/30"
          >
            <svg
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
              strokeWidth={1.5}
              className="w-full h-full"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
              />
            </svg>
          </motion.div>
          
          <motion.h3
            initial={{ opacity: 0, y: 15 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2, duration: 0.5 }}
            className="text-xl font-semibold text-white/90 mb-3 tracking-tight"
          >
            Results will appear here
          </motion.h3>
          
          <motion.p
            initial={{ opacity: 0, y: 15 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3, duration: 0.5 }}
            className="text-sm text-gray-400 mb-10 font-medium leading-relaxed tracking-wide"
          >
            Fill in the form and click analyze<br />
            to see predictions
          </motion.p>
          
          <motion.button
            type="button"
            onClick={onAnalyze}
            disabled={loading}
            whileHover={loading ? {} : { scale: 1.02, y: -2 }}
            whileTap={loading ? {} : { scale: 0.98, y: 0 }}
            className="px-8 py-3.5 bg-gradient-to-r from-blue-600 via-blue-500 to-blue-600 text-white text-base font-semibold rounded-xl hover:from-blue-500 hover:via-blue-400 hover:to-blue-500 transition-all duration-200 shadow-[0_4px_16px_rgba(59,130,246,0.4)] hover:shadow-[0_6px_24px_rgba(59,130,246,0.5)] disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {loading ? 'Processing...' : 'Analyze'}
          </motion.button>
        </div>
      </div>
    </motion.div>
  );
};
