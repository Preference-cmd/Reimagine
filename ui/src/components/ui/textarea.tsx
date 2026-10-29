import * as React from "react";

import { cn } from "@/lib/utils";

function Textarea({ className, ...props }: React.ComponentProps<"textarea">) {
  return (
    <textarea
      data-slot="textarea"
      className={cn(
        "flex field-sizing-content min-h-16 w-full rounded-md border border-hair bg-transparent px-3 py-2 text-base transition-[color,box-shadow] outline-none placeholder:text-fg3 focus-visible:border-accent focus-visible:ring-[3px] focus-visible:ring-accent/25 disabled:cursor-not-allowed disabled:opacity-50 aria-invalid:border-err aria-invalid:ring-err/20 md:text-sm",
        className,
      )}
      {...props}
    />
  );
}

export { Textarea };
