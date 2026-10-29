import * as React from "react";

import { cn } from "@/lib/utils";

function Input({ className, type, ...props }: React.ComponentProps<"input">) {
  return (
    <input
      type={type}
      data-slot="input"
      className={cn(
        "h-9 w-full min-w-0 rounded-md border border-hair bg-transparent px-3 py-1 text-base transition-[color,box-shadow] outline-none selection:bg-accent selection:text-on-accent file:inline-flex file:h-7 file:border-0 file:bg-transparent file:text-sm file:font-medium file:text-fg placeholder:text-fg3 disabled:pointer-events-none disabled:cursor-not-allowed disabled:opacity-50 md:text-sm",
        "focus-visible:border-accent focus-visible:ring-[3px] focus-visible:ring-accent/25",
        "aria-invalid:border-err aria-invalid:ring-err/20",
        className,
      )}
      {...props}
    />
  );
}

export { Input };
