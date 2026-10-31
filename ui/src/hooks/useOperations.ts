import { useEffect, useMemo, useState } from "react";
import { opList } from "@/ipc";
import type { OperationInfo } from "@/ipc/schemas";

/**
 * A descriptor is usable unless the host marked it unavailable. The health
 * shape is `{ status: "available" }` or `{ status: "unavailable", ... }`;
 * unknown shapes are treated as available so a newer host does not hide tools.
 */
export function isOperationAvailable(operation: OperationInfo | undefined): boolean {
  if (!operation) return false;
  const health = operation.health as { status?: unknown; state?: unknown } | null | undefined;
  if (!health || typeof health !== "object") return true;
  const status = health.status ?? health.state;
  return status === undefined || status === "available";
}

export type OperationsState = {
  operations: OperationInfo[];
  loading: boolean;
  error: string | null;
  textReplace: OperationInfo | undefined;
  textReplaceAvailable: boolean;
};

/** Load `op_list` for a project; the host reports mod loading failures there. */
export function useOperations(projectId: string): OperationsState {
  const [operations, setOperations] = useState<OperationInfo[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let active = true;
    setLoading(true);
    setError(null);
    opList(projectId).then(
      (list) => {
        if (!active) return;
        setOperations(Array.isArray(list) ? list : []);
        setLoading(false);
      },
      (err: unknown) => {
        if (!active) return;
        setOperations([]);
        setError(err instanceof Error ? err.message : String(err));
        setLoading(false);
      },
    );
    return () => {
      active = false;
    };
  }, [projectId]);

  const textReplace = useMemo(
    () => operations.find((operation) => operation.spec?.name === "text.replace"),
    [operations],
  );
  const textReplaceAvailable = isOperationAvailable(textReplace);

  return { operations, loading, error, textReplace, textReplaceAvailable };
}
