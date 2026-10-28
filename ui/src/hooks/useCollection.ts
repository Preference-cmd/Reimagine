import { useCallback, useEffect, useState } from "react";
import { opInvoke, subscribeEvents } from "@/ipc";
import { CollectionDocumentSchema, type CollectionDocument } from "@/ipc/schemas";

/**
 * Shared hook to load a collection document and keep it in sync with `collection.changed` events.
 */
export function useCollection(
  projectId: string,
  collectionId: string | null | undefined,
): {
  collection: CollectionDocument | null;
  loading: boolean;
  error: string | null;
  refresh: () => Promise<void>;
} {
  const [collection, setCollection] = useState<CollectionDocument | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchCollection = useCallback(async () => {
    if (!collectionId) {
      setCollection(null);
      setLoading(false);
      return;
    }
    setLoading(true);
    try {
      const res = await opInvoke(projectId, "collection.get", {
        collection_id: collectionId,
      });
      const output = CollectionDocumentSchema.parse(res.output);
      setCollection(output);
      setError(null);
    } catch (e) {
      setCollection(null);
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }, [projectId, collectionId]);

  useEffect(() => {
    void fetchCollection();
  }, [fetchCollection]);

  useEffect(() => {
    if (!collectionId) return;
    const unsubscribe = subscribeEvents((event) => {
      if (
        event.type === "document" &&
        event.kind === "collection.changed" &&
        event.projectId === projectId &&
        event.documentId === collectionId
      ) {
        void fetchCollection();
      }
    });
    return () => unsubscribe();
  }, [projectId, collectionId, fetchCollection]);

  return { collection, loading, error, refresh: fetchCollection };
}
