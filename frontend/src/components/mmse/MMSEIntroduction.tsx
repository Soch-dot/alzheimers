import React from 'react';
import { motion } from 'framer-motion';
import { GlassCard } from './primitives';

interface MMSEIntroductionProps {
  onStart: () => void;
}

export const MMSEIntroduction: React.FC<MMSEIntroductionProps> = ({ onStart }) => {
  return (
    <motion.div
      initial={{ opacity: 0, x: -24 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ duration: 0.5, ease: [0.16, 1, 0.3, 1] }}
    >
      <GlassCard className="text-center p-8 md:p-10">
        <h2 className="text-2xl md:text-3xl font-semibold text-white tracking-tight">
          MMSE Assessment
        </h2>
        <p className="text-sm text-gray-400 font-light tracking-wide mt-2">
          Mini-Mental State Examination
        </p>

        <div className="h-px w-24 mx-auto bg-gradient-to-r from-transparent via-white/20 to-transparent my-6" />

        <p className="text-sm text-gray-300 leading-relaxed max-w-sm mx-auto">
          This examiner-assisted assessment consists of 11 sections with a
          maximum score of 30 points.
        </p>
        <p className="text-xs text-gray-500 mt-4 max-w-sm mx-auto">
          The examiner asks the questions, observes the responses, and scores
          each item. This is a screening tool — it is not a diagnosis.
        </p>

        <motion.button
          type="button"
          onClick={onStart}
          whileHover={{ scale: 1.02, y: -2 }}
          whileTap={{ scale: 0.98, y: 0 }}
          className="mt-8 px-10 py-3.5 bg-gradient-to-r from-blue-600 via-blue-500 to-blue-600 text-white text-base font-semibold rounded-xl hover:from-blue-500 hover:via-blue-400 hover:to-blue-500 transition-all duration-200 shadow-[0_4px_16px_rgba(59,130,246,0.4)] hover:shadow-[0_6px_24px_rgba(59,130,246,0.5)]"
        >
          Start Assessment
        </motion.button>
      </GlassCard>
    </motion.div>
  );
};
