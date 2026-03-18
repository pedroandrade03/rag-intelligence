"use client";

import type { ChatStatus, FileUIPart } from "ai";
import type {
  ChangeEventHandler,
  ClipboardEventHandler,
  ComponentProps,
  FormEvent,
  FormEventHandler,
  HTMLAttributes,
  KeyboardEventHandler,
  MouseEvent,
} from "react";

import {
  InputGroup,
  InputGroupAddon,
  InputGroupButton,
  InputGroupTextarea,
} from "@/components/ui/input-group";
import { Spinner } from "@/components/ui/spinner";
import { cn } from "@/lib/utils";
import { CornerDownLeftIcon, SquareIcon, XIcon } from "lucide-react";
import { nanoid } from "nanoid";
import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";

interface PromptInputFile extends FileUIPart {
  id: string;
}

export interface AttachmentsContext {
  files: PromptInputFile[];
  add: (files: File[] | FileList) => void;
  clear: () => void;
  remove: (id: string) => void;
}

type PromptInputErrorCode = "accept" | "max_file_size" | "max_files";

interface PromptInputError {
  code: PromptInputErrorCode;
  message: string;
}

export interface PromptInputMessage {
  files: FileUIPart[];
  text: string;
}

export type PromptInputProps = Omit<ComponentProps<"form">, "onSubmit"> & {
  accept?: string;
  globalDrop?: boolean;
  maxFiles?: number;
  maxFileSize?: number;
  onError?: (error: PromptInputError) => void;
  onSubmit: (
    message: PromptInputMessage,
    event: FormEvent<HTMLFormElement>
  ) => void | Promise<void>;
};

const AttachmentsContext = createContext<AttachmentsContext | null>(null);

async function convertBlobUrlToDataUrl(url: string): Promise<string | null> {
  try {
    const response = await fetch(url);
    const blob = await response.blob();

    return await new Promise((resolve) => {
      const reader = new FileReader();
      reader.onloadend = () => resolve(reader.result as string);
      reader.onerror = () => resolve(null);
      reader.readAsDataURL(blob);
    });
  } catch {
    return null;
  }
}

function matchesAccept(file: File, accept?: string): boolean {
  if (!accept?.trim()) {
    return true;
  }

  return accept
    .split(",")
    .map((value) => value.trim())
    .filter(Boolean)
    .some((pattern) => {
      if (pattern.endsWith("/*")) {
        return file.type.startsWith(pattern.slice(0, -1));
      }

      return file.type === pattern;
    });
}

function toPromptInputFiles(files: File[]): PromptInputFile[] {
  return files.map((file) => ({
    filename: file.name,
    id: nanoid(),
    mediaType: file.type,
    type: "file",
    url: URL.createObjectURL(file),
  }));
}

function revokeFiles(files: PromptInputFile[]) {
  for (const file of files) {
    if (file.url) {
      URL.revokeObjectURL(file.url);
    }
  }
}

function selectFiles({
  accept,
  currentCount,
  files,
  maxFileSize,
  maxFiles,
  onError,
}: {
  accept?: string;
  currentCount: number;
  files: File[] | FileList;
  maxFileSize?: number;
  maxFiles?: number;
  onError?: (error: PromptInputError) => void;
}) {
  const incoming = Array.from(files);
  const accepted = incoming.filter((file) => matchesAccept(file, accept));

  if (incoming.length > 0 && accepted.length === 0) {
    onError?.({
      code: "accept",
      message: "No files match the accepted types.",
    });
    return [];
  }

  const sized = accepted.filter((file) =>
    maxFileSize ? file.size <= maxFileSize : true
  );

  if (accepted.length > 0 && sized.length === 0) {
    onError?.({
      code: "max_file_size",
      message: "All files exceed the maximum size.",
    });
    return [];
  }

  const capacity =
    typeof maxFiles === "number"
      ? Math.max(0, maxFiles - currentCount)
      : undefined;

  if (capacity === 0) {
    onError?.({
      code: "max_files",
      message: "Too many files. Some were not added.",
    });
    return [];
  }

  const capped = typeof capacity === "number" ? sized.slice(0, capacity) : sized;

  if (typeof capacity === "number" && sized.length > capacity) {
    onError?.({
      code: "max_files",
      message: "Too many files. Some were not added.",
    });
  }

  return capped;
}

export function usePromptInputAttachments() {
  const context = useContext(AttachmentsContext);

  if (!context) {
    throw new Error("usePromptInputAttachments must be used within PromptInput");
  }

  return context;
}

export function PromptInput({
  accept,
  children,
  className,
  globalDrop,
  maxFiles,
  maxFileSize,
  onError,
  onSubmit,
  ...props
}: PromptInputProps) {
  const [files, setFiles] = useState<PromptInputFile[]>([]);
  const filesRef = useRef<PromptInputFile[]>([]);
  const formRef = useRef<HTMLFormElement | null>(null);

  useEffect(() => {
    filesRef.current = files;
  }, [files]);

  useEffect(
    () => () => {
      revokeFiles(filesRef.current);
    },
    []
  );

  const add = useCallback(
    (incoming: File[] | FileList) => {
      const selected = selectFiles({
        accept,
        currentCount: filesRef.current.length,
        files: incoming,
        maxFileSize,
        maxFiles,
        onError,
      });

      if (selected.length === 0) {
        return;
      }

      const nextFiles = [...filesRef.current, ...toPromptInputFiles(selected)];
      filesRef.current = nextFiles;
      setFiles(nextFiles);
    },
    [accept, maxFileSize, maxFiles, onError]
  );

  const clear = useCallback(() => {
    revokeFiles(filesRef.current);
    filesRef.current = [];
    setFiles([]);
  }, []);

  const remove = useCallback((id: string) => {
    const nextFiles = filesRef.current.filter((file) => file.id !== id);
    revokeFiles(filesRef.current.filter((file) => file.id === id));
    filesRef.current = nextFiles;
    setFiles(nextFiles);
  }, []);

  useEffect(() => {
    const target = globalDrop ? document : formRef.current;

    if (!target) {
      return;
    }

    const onDragOver = (event: Event) => {
      if ((event as DragEvent).dataTransfer?.types?.includes("Files")) {
        event.preventDefault();
      }
    };

    const onDrop = (event: Event) => {
      const dragEvent = event as DragEvent;

      if (!dragEvent.dataTransfer?.types?.includes("Files")) {
        return;
      }

      event.preventDefault();

      if (dragEvent.dataTransfer.files.length > 0) {
        add(dragEvent.dataTransfer.files);
      }
    };

    target.addEventListener("dragover", onDragOver);
    target.addEventListener("drop", onDrop);

    return () => {
      target.removeEventListener("dragover", onDragOver);
      target.removeEventListener("drop", onDrop);
    };
  }, [add, globalDrop]);

  const attachments = useMemo<AttachmentsContext>(
    () => ({
      add,
      clear,
      files,
      remove,
    }),
    [add, clear, files, remove]
  );

  const handleSubmit: FormEventHandler<HTMLFormElement> = useCallback(
    async (event) => {
      event.preventDefault();

      const form = event.currentTarget;
      const text = ((new FormData(form).get("message") as string) ?? "").trim();

      try {
        const convertedFiles: FileUIPart[] = await Promise.all(
          filesRef.current.map(async ({ id: _id, ...file }) => {
            if (!file.url?.startsWith("blob:")) {
              return file;
            }

            const dataUrl = await convertBlobUrlToDataUrl(file.url);

            return {
              ...file,
              url: dataUrl ?? file.url,
            };
          })
        );

        const result = onSubmit({ files: convertedFiles, text }, event);

        if (result instanceof Promise) {
          await result;
        }

        form.reset();
        clear();
      } catch {
        // Keep the current draft when submission fails.
      }
    },
    [clear, onSubmit]
  );

  return (
    <AttachmentsContext.Provider value={attachments}>
      <form
        className="w-full"
        onSubmit={handleSubmit}
        ref={formRef}
        {...props}
      >
        <InputGroup className={cn("overflow-hidden", className)}>
          {children}
        </InputGroup>
      </form>
    </AttachmentsContext.Provider>
  );
}

export type PromptInputBodyProps = HTMLAttributes<HTMLDivElement>;

export function PromptInputBody({
  className,
  ...props
}: PromptInputBodyProps) {
  return <div className={cn("contents", className)} {...props} />;
}

export type PromptInputTextareaProps = ComponentProps<typeof InputGroupTextarea>;

export function PromptInputTextarea({
  className,
  onKeyDown,
  onPaste,
  placeholder = "What would you like to know?",
  ...props
}: PromptInputTextareaProps) {
  const attachments = usePromptInputAttachments();
  const [isComposing, setIsComposing] = useState(false);

  const handleKeyDown: KeyboardEventHandler<HTMLTextAreaElement> = useCallback(
    (event) => {
      onKeyDown?.(event);

      if (event.defaultPrevented) {
        return;
      }

      if (event.key === "Enter") {
        if (isComposing || event.nativeEvent.isComposing || event.shiftKey) {
          return;
        }

        event.preventDefault();

        const submitButton = event.currentTarget.form?.querySelector(
          'button[type="submit"]'
        ) as HTMLButtonElement | null;

        if (!submitButton?.disabled) {
          event.currentTarget.form?.requestSubmit();
        }
      }

      if (
        event.key === "Backspace" &&
        event.currentTarget.value === "" &&
        attachments.files.length > 0
      ) {
        event.preventDefault();
        const lastAttachment = attachments.files.at(-1);
        if (lastAttachment) {
          attachments.remove(lastAttachment.id);
        }
      }
    },
    [attachments, isComposing, onKeyDown]
  );

  const handlePaste: ClipboardEventHandler<HTMLTextAreaElement> = useCallback(
    (event) => {
      onPaste?.(event);

      if (event.defaultPrevented) {
        return;
      }

      const pastedFiles = Array.from(event.clipboardData?.items ?? [])
        .filter((item) => item.kind === "file")
        .map((item) => item.getAsFile())
        .filter((file): file is File => file !== null);

      if (pastedFiles.length === 0) {
        return;
      }

      event.preventDefault();
      attachments.add(pastedFiles);
    },
    [attachments, onPaste]
  );

  return (
    <InputGroupTextarea
      className={cn("field-sizing-content max-h-48 min-h-16", className)}
      name="message"
      onCompositionEnd={() => setIsComposing(false)}
      onCompositionStart={() => setIsComposing(true)}
      onKeyDown={handleKeyDown}
      onPaste={handlePaste}
      placeholder={placeholder}
      {...props}
    />
  );
}

export type PromptInputFooterProps = Omit<
  ComponentProps<typeof InputGroupAddon>,
  "align"
>;

export function PromptInputFooter({
  className,
  ...props
}: PromptInputFooterProps) {
  return (
    <InputGroupAddon
      align="block-end"
      className={cn("justify-between gap-1", className)}
      {...props}
    />
  );
}

export type PromptInputSubmitProps = ComponentProps<typeof InputGroupButton> & {
  onStop?: () => void;
  status?: ChatStatus;
};

export function PromptInputSubmit({
  children,
  className,
  onClick,
  onStop,
  size = "icon-sm",
  status,
  variant = "default",
  ...props
}: PromptInputSubmitProps) {
  const isGenerating = status === "submitted" || status === "streaming";

  let icon = <CornerDownLeftIcon className="size-4" />;

  if (status === "submitted") {
    icon = <Spinner />;
  } else if (status === "streaming") {
    icon = <SquareIcon className="size-4" />;
  } else if (status === "error") {
    icon = <XIcon className="size-4" />;
  }

  const handleClick = useCallback(
    (event: MouseEvent<HTMLButtonElement>) => {
      if (isGenerating && onStop) {
        event.preventDefault();
        onStop();
        return;
      }

      onClick?.(event);
    },
    [isGenerating, onClick, onStop]
  );

  return (
    <InputGroupButton
      aria-label={isGenerating ? "Stop" : "Submit"}
      className={cn(className)}
      onClick={handleClick}
      size={size}
      type={isGenerating && onStop ? "button" : "submit"}
      variant={variant}
      {...props}
    >
      {children ?? icon}
    </InputGroupButton>
  );
}
