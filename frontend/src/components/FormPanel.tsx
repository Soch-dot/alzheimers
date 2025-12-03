import React from 'react';
import { motion } from 'framer-motion';

interface FormPanelProps {
  children: React.ReactNode;
  onSubmit?: (e: React.FormEvent) => void;
}

export const FormPanel: React.FC<FormPanelProps> = ({ children, onSubmit }) => {
  return (
    <motion.div
      initial={{ opacity: 0, x: -40 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ duration: 0.6, ease: [0.16, 1, 0.3, 1] }}
      className="relative"
    >
      {/* Dark glassmorphism card */}
      <div className="relative bg-black/40 backdrop-blur-2xl rounded-[2rem] shadow-[0_8px_32px_rgba(0,0,0,0.4)] border border-white/10 p-10 md:p-12 overflow-hidden">
        {/* Gradient overlay layers */}
        <div className="absolute inset-0 bg-gradient-to-br from-white/5 via-transparent to-black/20 pointer-events-none" />
        <div className="absolute inset-0 bg-gradient-to-t from-blue-500/5 via-transparent to-transparent pointer-events-none" />
        
        {/* Subtle inner glow */}
        <div className="absolute inset-[1px] rounded-[calc(2rem-1px)] bg-gradient-to-b from-white/10 to-transparent pointer-events-none" />
        
        {/* Content */}
        <div className="relative">
          <form onSubmit={onSubmit} className="space-y-10">
            {children}
          </form>
        </div>
      </div>
    </motion.div>
  );
};
