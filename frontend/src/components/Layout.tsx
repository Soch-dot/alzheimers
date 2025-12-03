import React from 'react';
import { motion } from 'framer-motion';

interface LayoutProps {
  children: React.ReactNode;
}

export const Layout: React.FC<LayoutProps> = ({ children }) => {
  return (
    <div className="min-h-screen relative overflow-hidden bg-gradient-to-br from-gray-950 via-black to-gray-900">
      {/* Dark gradient layers for depth */}
      <div className="fixed inset-0 bg-gradient-to-br from-gray-950/90 via-black to-gray-900/90 -z-10" />
      <div className="fixed inset-0 bg-[radial-gradient(ellipse_at_top_right,_var(--tw-gradient-stops))] from-blue-900/20 via-transparent to-transparent -z-10" />
      <div className="fixed inset-0 bg-[radial-gradient(ellipse_at_bottom_left,_var(--tw-gradient-stops))] from-purple-900/10 via-transparent to-transparent -z-10" />
      
      {/* Subtle grid pattern */}
      <div className="fixed inset-0 bg-[linear-gradient(to_right,#ffffff05_1px,transparent_1px),linear-gradient(to_bottom,#ffffff05_1px,transparent_1px)] bg-[size:32px_32px] -z-10 opacity-20" />
      
      <div className="relative max-w-7xl mx-auto px-5 sm:px-6 lg:px-10 py-24 md:py-32">
        {/* Header with refined typography */}
        <motion.div
          initial={{ opacity: 0, y: -40 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.7, ease: [0.16, 1, 0.3, 1] }}
          className="mb-24 md:mb-28 text-center"
        >
          <h1 className="text-5xl md:text-6xl lg:text-7xl font-semibold text-white mb-6 tracking-[-0.02em] leading-[1.1]">
            Alzheimer's Risk
            <br />
            <span className="bg-gradient-to-r from-white via-blue-100 to-white bg-clip-text text-transparent">
              Assessment
            </span>
          </h1>
          <motion.p
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2, duration: 0.6 }}
            className="text-lg md:text-xl text-gray-400 font-light tracking-wide mb-10"
          >
            Clinical data analysis using machine learning
          </motion.p>
          <motion.div
            initial={{ opacity: 0, scaleX: 0 }}
            animate={{ opacity: 1, scaleX: 1 }}
            transition={{ delay: 0.4, duration: 0.6 }}
            className="flex items-center justify-center gap-3"
          >
            <div className="h-px w-20 bg-gradient-to-r from-transparent via-white/20 to-transparent" />
            <div className="w-1.5 h-1.5 rounded-full bg-blue-400/60" />
            <div className="h-px w-20 bg-gradient-to-l from-transparent via-white/20 to-transparent" />
          </motion.div>
        </motion.div>

        {children}
      </div>
    </div>
  );
};
