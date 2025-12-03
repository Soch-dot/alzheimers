import React from 'react';
import { motion } from 'framer-motion';

interface AnalyzeButtonProps {
  onClick: () => void;
  loading: boolean;
  disabled?: boolean;
}

export const AnalyzeButton: React.FC<AnalyzeButtonProps> = ({
  onClick,
  loading,
  disabled = false,
}) => {
  return (
    <motion.button
      type="button"
      onClick={onClick}
      disabled={loading || disabled}
      whileHover={disabled || loading ? {} : { scale: 1.01, y: -2 }}
      whileTap={disabled || loading ? {} : { scale: 0.99, y: 0 }}
      className="w-full px-8 py-4.5 bg-gradient-to-r from-blue-600 via-blue-500 to-blue-600 text-white text-base font-semibold rounded-xl hover:from-blue-500 hover:via-blue-400 hover:to-blue-500 active:from-blue-700 active:via-blue-600 active:to-blue-700 transition-all duration-200 shadow-[0_4px_16px_rgba(59,130,246,0.4)] hover:shadow-[0_6px_24px_rgba(59,130,246,0.5)] disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:from-blue-600 disabled:hover:via-blue-500 disabled:hover:to-blue-600 disabled:hover:shadow-[0_4px_16px_rgba(59,130,246,0.4)] relative overflow-hidden"
    >
      {/* Shimmer effect */}
      <motion.div
        className="absolute inset-0 bg-gradient-to-r from-transparent via-white/20 to-transparent"
        initial={{ x: '-100%' }}
        whileHover={disabled || loading ? {} : { x: '100%' }}
        transition={{ duration: 0.7, ease: 'easeInOut' }}
      />
      
      <span className="relative flex items-center justify-center gap-3">
        {loading ? (
          <>
            <svg
              className="animate-spin h-5 w-5"
              xmlns="http://www.w3.org/2000/svg"
              fill="none"
              viewBox="0 0 24 24"
            >
              <circle
                className="opacity-25"
                cx="12"
                cy="12"
                r="10"
                stroke="currentColor"
                strokeWidth="4"
              />
              <path
                className="opacity-75"
                fill="currentColor"
                d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
              />
            </svg>
            Processing...
          </>
        ) : (
          'Analyze'
        )}
      </span>
    </motion.button>
  );
};
