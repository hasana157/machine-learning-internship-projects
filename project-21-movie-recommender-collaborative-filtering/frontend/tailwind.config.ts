import type { Config } from 'tailwindcss';
import defaultTheme from 'tailwindcss/defaultTheme';

const config: Config = {
  content: [
    './app/**/*.{js,ts,jsx,tsx,mdx}',
    './components/**/*.{js,ts,jsx,tsx,mdx}',
  ],

  theme: {
    extend: {
      colors: {
        // Cinematic dark theme - matching design system
        'bg-primary': '#0A0E1A',
        'bg-card': '#111827',
        'bg-elevated': '#0D1A30',
        'accent-blue': '#1A6CF6',
        'accent-cyan': '#00D4FF',
        'accent-purple': '#7B2FFF',
        'accent-gold': '#FFB800',
        'text-primary': '#F0F4FF',
        'text-secondary': '#8A93A8',
        'border-subtle': '#1A2A4A',

        // Standard palette
        'dark': {
          50: '#F9FAFB',
          100: '#F3F4F6',
          200: '#E5E7EB',
          300: '#D1D5DB',
          400: '#9CA3AF',
          500: '#6B7280',
          600: '#4B5563',
          700: '#374151',
          800: '#1F2937',
          900: '#111827',
          950: '#030712',
        },
      },

      fontFamily: {
        sans: ['Inter', ...defaultTheme.fontFamily.sans],
        mono: ['JetBrains Mono', ...defaultTheme.fontFamily.mono],
      },

      fontSize: {
        // Custom type scale
        'display': ['48px', { lineHeight: '60px', fontWeight: '800' }],
        'h1': ['32px', { lineHeight: '40px', fontWeight: '700' }],
        'h2': ['24px', { lineHeight: '32px', fontWeight: '600' }],
        'h3': ['18px', { lineHeight: '28px', fontWeight: '600' }],
        'body-lg': ['16px', { lineHeight: '24px', fontWeight: '400' }],
        'body': ['14px', { lineHeight: '20px', fontWeight: '400' }],
        'caption': ['12px', { lineHeight: '16px', fontWeight: '400' }],
      },

      animation: {
        'fade-in': 'fadeIn 0.3s ease-in',
        'slide-up': 'slideUp 0.3s ease-out',
        'scale-in': 'scaleIn 0.2s ease-out',
        'pulse-subtle': 'pulseSubtle 2s cubic-bezier(0.4, 0, 0.6, 1) infinite',
      },

      keyframes: {
        fadeIn: {
          '0%': { opacity: '0' },
          '100%': { opacity: '1' },
        },
        slideUp: {
          '0%': { transform: 'translateY(10px)', opacity: '0' },
          '100%': { transform: 'translateY(0)', opacity: '1' },
        },
        scaleIn: {
          '0%': { transform: 'scale(0.95)', opacity: '0' },
          '100%': { transform: 'scale(1)', opacity: '1' },
        },
        pulseSubtle: {
          '0%, 100%': { opacity: '1' },
          '50%': { opacity: '.8' },
        },
      },

      backdropFilter: {
        'blur': 'blur(20px)',
      },

      boxShadow: {
        'card': '0 4px 20px rgba(0, 0, 0, 0.3)',
        'hover': '0 8px 40px rgba(26, 108, 246, 0.2)',
        'glow': '0 0 20px rgba(26, 108, 246, 0.4)',
      },

      spacing: {
        'safe-top': 'env(safe-area-inset-top)',
        'safe-bottom': 'env(safe-area-inset-bottom)',
        'safe-left': 'env(safe-area-inset-left)',
        'safe-right': 'env(safe-area-inset-right)',
      },
    },
  },

  plugins: [
    // Typography plugin for prose
    require('@tailwindcss/typography'),
  ],
};

export default config;
