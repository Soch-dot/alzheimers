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
  error?: string | null;
}

export const SelectField: React.FC<SelectFieldProps> = ({
  label,
  name,
  value,
  onChange,
  options,
  required = false,
  error,
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
          aria-invalid={error ? true : undefined}
          className={`w-full px-4 py-3.5 text-base text-white bg-white/5 backdrop-blur-sm border rounded-xl focus:ring-2 transition-all duration-200 shadow-[0_2px_8px_rgba(0,0,0,0.3)] hover:shadow-[0_4px_12px_rgba(0,0,0,0.4)] cursor-pointer appearance-none ${
            error
              ? 'border-rose-500/60 focus:ring-rose-400/30 focus:border-rose-400/70'
              : 'border-white/10 focus:ring-blue-400/30 focus:border-blue-400/50 hover:border-white/20 focus:bg-white/10'
          }`}
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
      {error && (
        <p className="text-xs text-rose-400 font-medium mt-2 tracking-wide">{error}</p>
      )}
    </motion.div>
  );
};
