import React from 'react';
import { motion } from 'framer-motion';

interface InputFieldProps {
  label: string;
  name: string;
  type?: 'number' | 'text';
  value: number | string;
  onChange: (e: React.ChangeEvent<HTMLInputElement>) => void;
  min?: number;
  max?: number;
  step?: number;
  required?: boolean;
  hint?: string;
}

export const InputField: React.FC<InputFieldProps> = ({
  label,
  name,
  type = 'number',
  value,
  onChange,
  min,
  max,
  step,
  required = false,
  hint,
}) => {
  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
      className="space-y-3"
    >
      <label className="block text-xs font-semibold text-gray-400 uppercase tracking-[0.08em]">
        {label}
      </label>
      <div className="relative">
        <input
          type={type}
          name={name}
          value={value}
          onChange={onChange}
          className="w-full px-4 py-3.5 text-base text-white bg-white/5 backdrop-blur-sm border border-white/10 rounded-xl focus:ring-2 focus:ring-blue-400/30 focus:border-blue-400/50 focus:bg-white/10 transition-all duration-200 shadow-[0_2px_8px_rgba(0,0,0,0.3)] hover:shadow-[0_4px_12px_rgba(0,0,0,0.4)] hover:border-white/20 placeholder:text-gray-500"
          required={required}
          min={min}
          max={max}
          step={step}
        />
      </div>
      {hint && (
        <p className="text-xs text-gray-500 font-medium mt-2 tracking-wide">{hint}</p>
      )}
    </motion.div>
  );
};
