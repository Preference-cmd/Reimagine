import { useCallback, useEffect, useMemo, useState } from "react";
import { opList } from "@/ipc";
import type { OperationInfo } from "@/ipc/schemas";

/** One automatic retry covers a host that is still starting its mods. */
const OPERATIONS_RETRY_DELAY_MS = 400;

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
  /** Reload `op_list`; the caller surfaces this when the automatic retry fails. */
  retry: () => void;
};

/**
 * Load `op_list` for a project; the host reports mod loading failures there.
 *
 * A failed read is retried once, because the host may still be starting its
 * mods. A second failure is exposed through `error`, and `retry` lets the caller
 * offer another attempt instead of hiding the tools forever.
 */
export function useOperations(projectId: string): OperationsState {
  const [operations, setOperations] = useState<OperationInfo[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [reloadToken, setReloadToken] = useState(0);

  useEffect(() => {
    let active = true;
    let retried = false;
    let retryTimer: ReturnType<typeof setTimeout> | undefined;

    setLoading(true);
    setError(null);

    const load = () => {
      opList(projectId).then(
        (list) => {
          if (!active) return;
          setOperations(Array.isArray(list) ? list : []);
          setLoading(false);
        },
        (err: unknown) => {
          if (!active) return;
          if (!retried) {
            retried = true;
            retryTimer = setTimeout(load, OPERATIONS_RETRY_DELAY_MS);
            return;
          }
          setOperations([]);
          setError(err instanceof Error ? err.message : String(err));
          setLoading(false);
        },
      );
    };
    load();

    return () => {
      active = false;
      if (retryTimer !== undefined) clearTimeout(retryTimer);
    };
  }, [projectId, reloadToken]);

  const retry = useCallback(() => setReloadToken((token) => token + 1), []);

  const textReplace = useMemo(
    () => operations.find((operation) => operation.spec?.name === "text.replace"),
    [operations],
  );
  const textReplaceAvailable = isOperationAvailable(textReplace);

  return { operations, loading, error, textReplace, textReplaceAvailable, retry };
}
