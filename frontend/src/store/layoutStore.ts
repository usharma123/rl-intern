import { create } from 'zustand';
import { persist } from 'zustand/middleware';

const DEFAULT_MODEL = 'openrouter/anthropic/claude-sonnet-4.5';

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
    { name: 'rl-intern-layout' },
  ),
);
