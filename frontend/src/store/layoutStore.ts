import { create } from 'zustand';
import { persist } from 'zustand/middleware';

const DEFAULT_MODEL = 'openrouter/openai/gpt-oss-120b:free';
const PREVIOUS_DEFAULT_MODELS = new Set([
  'openrouter/anthropic/claude-sonnet-4.5',
  'anthropic/claude-sonnet-4-5-20250929',
  'nvidia/nemotron-3-nano-30b-a3b:free',
  'openrouter/nvidia/nemotron-3-nano-30b-a3b:free',
  'google/gemma-4-26b-a4b-it:free',
  'openrouter/google/gemma-4-26b-a4b-it:free',
  'openai/gpt-oss-120b:free',
]);

interface LayoutState {
  isLeftSidebarOpen: boolean;
  model: string;
  toggleLeftSidebar: () => void;
  setLeftSidebarOpen: (open: boolean) => void;
  setModel: (model: string) => void;
}

export const useLayoutStore = create<LayoutState>()(
  persist(
    (set) => ({
      isLeftSidebarOpen: true,
      model: DEFAULT_MODEL,
      toggleLeftSidebar: () => set((s) => ({ isLeftSidebarOpen: !s.isLeftSidebarOpen })),
      setLeftSidebarOpen: (open) => set({ isLeftSidebarOpen: open }),
      setModel: (model) => set({ model: model.trim() || DEFAULT_MODEL }),
    }),
    {
      name: 'rl-intern-layout',
      version: 4,
      migrate: (persistedState) => {
        if (
          persistedState &&
          typeof persistedState === 'object' &&
          'model' in persistedState &&
          typeof persistedState.model === 'string' &&
          PREVIOUS_DEFAULT_MODELS.has(persistedState.model)
        ) {
          return { ...persistedState, model: DEFAULT_MODEL };
        }
        return persistedState;
      },
    },
  ),
);
