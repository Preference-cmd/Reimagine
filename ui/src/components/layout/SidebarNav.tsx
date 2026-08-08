import {
  ChevronDown,
  ChevronRight,
  Clock,
  FolderOpen,
  GitBranch,
  PenSquare,
  Puzzle,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { useUIStore, type SidebarSection } from "@/store/uiStore";
import { useWorkflowStore } from "@/store/workflow";
import * as m from "$paraglide/messages";

/* ───── Primary nav items (Codex pattern) ───── */

type PrimaryNavItem = {
  icon: React.ComponentType<{ className?: string }>;
  labelKey: string;
  section: SidebarSection;
};

const PRIMARY_NAV_ITEMS: PrimaryNavItem[] = [
  { icon: PenSquare, labelKey: "sidebar.newTask", section: "new-task" },
  { icon: GitBranch, labelKey: "sidebar.pullRequests", section: "pull-requests" },
  { icon: Clock, labelKey: "sidebar.scheduled", section: "scheduled" },
  { icon: Puzzle, labelKey: "sidebar.plugins", section: "plugins" },
];

/* ───── Projects section ───── */

type ProjectItem = {
  id: string;
  name: string;
  subText?: string;
  onClick?: () => void;
};

export function SidebarNav() {
  const activeSection = useUIStore((s) => s.activeSidebarSection);
  const setActiveSection = useUIStore((s) => s.setActiveSidebarSection);
  const projectsExpanded = useUIStore((s) => s.projectsExpanded);
  const setProjectsExpanded = useUIStore((s) => s.setProjectsExpanded);
  const activeProjectId = useUIStore((s) => s.activeProjectId);
  const setActiveProjectId = useUIStore((s) => s.setActiveProjectId);

  const workflowName = useWorkflowStore((s) => s.name);

  const projects: ProjectItem[] = [
    {
      id: "current",
      name: workflowName,
      subText: "Active workflow",
      onClick: () => {
        setActiveProjectId("current");
        setActiveSection("workflows");
      },
    },
  ];

  const navigateToPrimary = (section: SidebarSection) => {
    setActiveSection(section);
    setActiveProjectId(null);
  };

  return (
    <nav className="flex flex-1 flex-col gap-0 px-2 pt-1" aria-label="Sidebar navigation">
      {/* Primary navigation items */}
      <div className="space-y-0.5">
        {PRIMARY_NAV_ITEMS.map((item) => (
          <NavButton
            key={item.section}
            item={item}
            active={activeSection === item.section}
            onClick={() => navigateToPrimary(item.section)}
          />
        ))}
      </div>

      {/* Divider */}
      <div className="my-3 h-px bg-sidebar-border" />

      {/* Projects section */}
      <div className="px-1">
        <button
          type="button"
          onClick={() => setProjectsExpanded(!projectsExpanded)}
          className="flex w-full items-center gap-1.5 py-1 text-left text-[11px] font-semibold uppercase tracking-wider text-sidebar-section-header transition-colors hover:text-sidebar-text-secondary"
        >
          {projectsExpanded ? (
            <ChevronDown className="h-3 w-3" />
          ) : (
            <ChevronRight className="h-3 w-3" />
          )}
          {m["sidebar.projects"]()}
        </button>

        {projectsExpanded && (
          <div className="mt-0.5 space-y-0.5">
            {projects.map((project) => (
              <ProjectRow
                key={project.id}
                project={project}
                active={activeProjectId === project.id}
                onClick={() => project.onClick?.()}
              />
            ))}
          </div>
        )}
      </div>

      {/* Tasks section */}
      <div className="mt-3 px-1">
        <div className="py-1 text-[11px] font-semibold uppercase tracking-wider text-sidebar-section-header">
          {m["sidebar.tasks"]()}
        </div>
        <div className="mt-1 rounded-lg bg-sidebar-item-hover px-3 py-4 text-center text-caption text-sidebar-text-muted">
          {m["sidebar.noTasks"]()}
        </div>
      </div>
    </nav>
  );
}

/* ───── Sub-components ───── */

function NavButton({
  item,
  active,
  onClick,
}: {
  item: PrimaryNavItem;
  active: boolean;
  onClick: () => void;
}) {
  const Icon = item.icon;
  const label = (m as unknown as Record<string, () => string>)[item.labelKey]();

  return (
    <button
      type="button"
      aria-current={active ? "page" : undefined}
      onClick={onClick}
      className={cn(
        "flex h-8 w-full cursor-pointer items-center gap-2.5 rounded-md px-2.5 text-left text-sm transition-colors duration-150",
        "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-white/20",
        active
          ? "bg-sidebar-item-active text-sidebar-text-primary"
          : "text-sidebar-text-secondary hover:bg-sidebar-item-hover hover:text-sidebar-text-primary",
      )}
    >
      <Icon
        className={cn(
          "h-4 w-4 shrink-0",
          active ? "text-sidebar-text-secondary" : "text-sidebar-text-muted",
        )}
      />
      <span className="truncate">{label}</span>
    </button>
  );
}

function ProjectRow({
  project,
  active,
  onClick,
}: {
  project: ProjectItem;
  active: boolean;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={cn(
        "flex w-full items-center gap-2 rounded-md px-2 py-1.5 text-left transition-colors",
        "hover:bg-sidebar-item-hover",
        active && "bg-sidebar-item-active",
      )}
    >
      <FolderOpen
        className={cn(
          "h-4 w-4 shrink-0",
          active ? "text-sidebar-text-primary" : "text-sidebar-text-muted",
        )}
      />
      <div className="min-w-0 flex-1">
        <div
          className={cn(
            "truncate text-sm",
            active ? "text-sidebar-text-primary" : "text-sidebar-text-secondary",
          )}
        >
          {project.name}
        </div>
        {project.subText && (
          <div className="truncate text-caption text-sidebar-text-muted">{project.subText}</div>
        )}
      </div>
    </button>
  );
}
