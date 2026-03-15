"use client";

import { useEffect } from "react";

export function DevTools() {
  useEffect(() => {
    if (process.env.NODE_ENV !== "development") {
      return;
    }

    void import("react-grab");
  }, []);

  return null;
}
