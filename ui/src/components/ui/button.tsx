import * as React from "react";
import { cva, type VariantProps } from "class-variance-authority";
import { Slot } from "radix-ui";

import { cn } from "@/lib/utils";

const buttonVariants = cva(
  "inline-flex shrink-0 items-center justify-center gap-2 rounded-full text-sm font-medium whitespace-nowrap transition-[color,background-color,box-shadow,transform] duration-150 ease-ui outline-none focus-visible:ring-2 focus-visible:ring-accent/30 active:scale-[0.97] disabled:pointer-events-none disabled:opacity-50 aria-invalid:border-err aria-invalid:ring-err/20 [&_svg]:pointer-events-none [&_svg]:shrink-0 [&_svg:not([class*='size-'])]:size-4",
  {
    variants: {
      variant: {
        default: "bg-accent text-on-accent shadow-1 hover:bg-accent-hover",
        secondary: "border border-hair-strong bg-surface text-fg shadow-1 hover:bg-hover",
        outline: "border border-hair bg-transparent text-fg hover:bg-hover",
        destructive: "bg-err text-on-accent hover:bg-err/90",
        ghost: "text-fg2 hover:bg-hover hover:text-fg active:bg-press",
        link: "text-info underline-offset-4 hover:underline",
      },
      size: {
        default: "h-10 px-4 py-2 text-body-sm has-[>svg]:px-3",
        xs: "h-7 gap-1 px-2 text-xs has-[>svg]:px-1.5 [&_svg:not([class*='size-'])]:size-3",
        sm: "h-8 gap-1.5 px-3 text-body-sm has-[>svg]:px-2.5",
        lg: "h-12 px-6 text-body-md has-[>svg]:px-4",
        icon: "size-10",
        "icon-xs": "size-7 [&_svg:not([class*='size-'])]:size-3",
        "icon-sm": "size-8",
        "icon-lg": "size-12",
      },
    },
    defaultVariants: {
      variant: "default",
      size: "default",
    },
  },
);

function Button({
  className,
  variant = "default",
  size = "default",
  asChild = false,
  ...props
}: React.ComponentProps<"button"> &
  VariantProps<typeof buttonVariants> & {
    asChild?: boolean;
  }) {
  const Comp = asChild ? Slot.Root : "button";

  return (
    <Comp
      data-slot="button"
      data-variant={variant}
      data-size={size}
      className={cn(buttonVariants({ variant, size, className }))}
      {...props}
    />
  );
}

export { Button, buttonVariants };
