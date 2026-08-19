import { create } from "zustand";
import { listProjects, setActiveProject, loadProject } from "@/ipc";
import type { Project } from "@/ipc/schemas";

type ProjectState = {
  projects: Project[];
  activeProjectId: string;
  loading: boolean;
  hydrate: (projects: Project[]) => void;
  load: () => Promise<void>;
  switchProject: (projectId: string) => Promise<Project>;
};

export const useProjectStore = create<ProjectState>((set, get) => ({
  projects: [],
  activeProjectId: "default",
  loading: false,
  hydrate: (projects) => {
    const active = get().activeProjectId;
    set({ projects, activeProjectId: projects.some((p) => p.id === active) ? active : (projects[0]?.id ?? "default") });
  },
  load: async () => {
    set({ loading: true });
    try { set({ projects: await listProjects() }); } finally { set({ loading: false }); }
  },
  switchProject: async (projectId) => {
    const project = await setActiveProject(projectId);
    set({ activeProjectId: project.id });
    await loadProject(project.id);
    return project;
  },
}));