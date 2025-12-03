import React from 'react';
import { motion } from 'framer-motion';

interface SelectOption {
  value: number | string;
  label: string;
}

interface SelectFieldProps {
  label: string;
  name: string;
  value: number | string;
  onChange: (e: React.ChangeEvent<HTMLSelectElement>) => void;
  options: SelectOption[];
  required?: boolean;
}

export const SelectField: React.FC<SelectFieldProps> = ({
  label,
  name,
  value,
  onChange,
  options,
  required = false,
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
        <select
          name={name}
          value={value}
          onChange={onChange}
          className="w-full px-4 py-3.5 text-base text-white bg-white/5 backdrop-blur-sm border border-white/10 rounded-xl focus:ring-2 focus:ring-blue-400/30 focus:border-blue-400/50 focus:bg-white/10 transition-all duration-200 shadow-[0_2px_8px_rgba(0,0,0,0.3)] hover:shadow-[0_4px_12px_rgba(0,0,0,0.4)] hover:border-white/20 cursor-pointer appearance-none"
          required={required}
          style={{
            backgroundImage: `url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' fill='none' viewBox='0 0 24 24' stroke='%23ffffff'%3E%3Cpath stroke-linecap='round' stroke-linejoin='round' stroke-width='2' d='M19 9l-7 7-7-7'%3E%3C/path%3E%3C/svg%3E")`,
            backgroundRepeat: 'no-repeat',
            backgroundPosition: 'right 1rem center',
            backgroundSize: '1.25rem',
            paddingRight: '3rem',
          }}
        >
          {options.map((option) => (
            <option key={option.value} value={option.value} className="bg-gray-900 text-white">
              {option.label}
            </option>
          ))}
        </select>
      </div>
    </motion.div>
  );
};
