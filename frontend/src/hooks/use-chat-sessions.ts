"use client";

import type { UIMessage } from "ai";

import {
  useMutation,
  useQuery,
  useQueryClient,
} from "@tanstack/react-query";
import { useCallback, useEffect, useRef, useState } from "react";

import {
  createSession,
  deleteSession,
  listSessionMessages,
  listSessions,
  persistSessionMessages,
  sessionKeys,
  updateSessionTitle,
} from "@/lib/chat-session-api";
import { DEFAULT_CHAT_TITLE, type ChatSession } from "@/lib/chat-sessions";

interface UseChatSessionsOptions {
  replaceMessages: (messages: UIMessage[]) => void;
}

const EMPTY_SESSIONS: ChatSession[] = [];

export function useChatSessions({
  replaceMessages,
}: UseChatSessionsOptions) {
  const queryClient = useQueryClient();
  const [selectedChatId, setSelectedChatId] = useState<string | null>(null);
  const bootstrappingRef = useRef(false);

  const sessionsQuery = useQuery({
    queryFn: listSessions,
    queryKey: sessionKeys.list(),
  });

  const chatSessions = sessionsQuery.data ?? EMPTY_SESSIONS;

  const activeChatId =
    selectedChatId && chatSessions.some((session) => session.id === selectedChatId)
      ? selectedChatId
      : chatSessions[0]?.id ?? null;

  const messagesQuery = useQuery({
    enabled: !!activeChatId,
    queryFn: () => listSessionMessages(activeChatId!),
    queryKey: activeChatId
      ? sessionKeys.messages(activeChatId)
      : ["sessions", "messages", "idle"],
  });

  const createSessionMutation = useMutation({
    mutationFn: createSession,
    mutationKey: ["sessions", "create"],
    onSuccess: (session) => {
      queryClient.setQueryData<ChatSession[]>(
        sessionKeys.list(),
        (current = []) => [session, ...current]
      );
      queryClient.setQueryData<UIMessage[]>(sessionKeys.messages(session.id), []);
    },
  });

  const updateTitleMutation = useMutation({
    mutationFn: updateSessionTitle,
    mutationKey: ["sessions", "update-title"],
    onSuccess: ({ id, title }) => {
      queryClient.setQueryData<ChatSession[]>(
        sessionKeys.list(),
        (current = []) =>
          current.map((session) =>
            session.id === id ? { ...session, title } : session
          )
      );
    },
  });

  const deleteSessionMutation = useMutation({
    mutationFn: deleteSession,
    mutationKey: ["sessions", "delete"],
    onSuccess: (id) => {
      queryClient.setQueryData<ChatSession[]>(
        sessionKeys.list(),
        (current = []) => current.filter((session) => session.id !== id)
      );
      queryClient.removeQueries({ queryKey: sessionKeys.messages(id) });
    },
  });

  const persistMessagesMutation = useMutation({
    mutationFn: persistSessionMessages,
    mutationKey: ["sessions", "persist-messages"],
    onSuccess: (messages, variables) => {
      queryClient.setQueryData(sessionKeys.messages(variables.id), messages);
    },
  });

  useEffect(() => {
    if (!sessionsQuery.isSuccess || chatSessions.length > 0) {
      return;
    }

    if (bootstrappingRef.current || createSessionMutation.isPending) {
      return;
    }

    bootstrappingRef.current = true;

    createSessionMutation.mutate(DEFAULT_CHAT_TITLE, {
      onError: () => {
        bootstrappingRef.current = false;
      },
      onSuccess: (session) => {
        setSelectedChatId(session.id);
        replaceMessages([]);
      },
    });
  }, [
    chatSessions.length,
    createSessionMutation,
    replaceMessages,
    sessionsQuery.isSuccess,
  ]);

  useEffect(() => {
    if (!activeChatId) {
      replaceMessages([]);
      return;
    }

    const cachedMessages = queryClient.getQueryData<UIMessage[]>(
      sessionKeys.messages(activeChatId)
    );

    replaceMessages(cachedMessages ?? []);
  }, [activeChatId, queryClient, replaceMessages]);

  useEffect(() => {
    if (messagesQuery.isSuccess) {
      replaceMessages(messagesQuery.data ?? []);
    }
  }, [messagesQuery.data, messagesQuery.isSuccess, replaceMessages]);

  const selectChat = useCallback(
    (id: string) => {
      if (id === activeChatId) {
        return;
      }

      setSelectedChatId(id);
    },
    [activeChatId]
  );

  const createNewChat = useCallback(async () => {
    const session = await createSessionMutation.mutateAsync(DEFAULT_CHAT_TITLE);
    setSelectedChatId(session.id);
    replaceMessages([]);
    return session;
  }, [createSessionMutation, replaceMessages]);

  const deleteChatById = useCallback(
    async (id: string) => {
      const remainingSessions = chatSessions.filter((session) => session.id !== id);

      try {
        await deleteSessionMutation.mutateAsync(id);
      } catch (error) {
        console.error("Failed to delete session", error);
        return;
      }

      if (remainingSessions.length === 0) {
        const session = await createSessionMutation.mutateAsync(DEFAULT_CHAT_TITLE);
        setSelectedChatId(session.id);
        replaceMessages([]);
        return;
      }

      if (id === activeChatId) {
        setSelectedChatId(remainingSessions[0].id);
      }
    },
    [
      activeChatId,
      chatSessions,
      createSessionMutation,
      deleteSessionMutation,
      replaceMessages,
    ]
  );

  const renameActiveSession = useCallback(
    async (title: string) => {
      if (!activeChatId) {
        return;
      }

      const nextTitle = title.trim() || DEFAULT_CHAT_TITLE;

      try {
        await updateTitleMutation.mutateAsync({
          id: activeChatId,
          title: nextTitle,
        });
      } catch (error) {
        console.error("Failed to update session title", error);
      }
    },
    [activeChatId, updateTitleMutation]
  );

  const persistMessages = useCallback(
    async (id: string, messages: UIMessage[]) => {
      try {
        await persistMessagesMutation.mutateAsync({ id, messages });
      } catch (error) {
        console.error("Failed to persist session messages", error);
      }
    },
    [persistMessagesMutation]
  );

  const loaded =
    !sessionsQuery.isPending &&
    !(sessionsQuery.isSuccess && chatSessions.length === 0 && createSessionMutation.isPending);

  return {
    activeChatId,
    chatSessions,
    createNewChat,
    deleteChat: deleteChatById,
    isLoadingMessages: messagesQuery.isPending,
    loaded,
    persistMessages,
    selectChat,
    updateSessionTitle: renameActiveSession,
  };
}
