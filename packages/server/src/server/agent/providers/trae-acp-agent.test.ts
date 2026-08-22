import {
  RequestError,
  type PromptResponse,
  type RequestPermissionRequest,
} from "@agentclientprotocol/sdk";
import { expect, test, vi } from "vitest";

import { createTestLogger } from "../../../test-utils/test-logger.js";
import { asInternals } from "../../test-utils/class-mocks.js";
import { DEFAULT_ACP_CAPABILITIES } from "./acp-agent.js";
import { TraeACPAgentSession } from "./trae-acp-agent.js";

const ROOT_THREAD_ID = "parent-thread";

interface TraeSessionInternals {
  sessionId: string | null;
  activeForegroundTurnId: string | null;
  connection: {
    prompt: (...args: unknown[]) => Promise<PromptResponse>;
    cancel?: (...args: unknown[]) => Promise<void>;
    extMethod?: (
      method: string,
      params: Record<string, unknown>,
    ) => Promise<Record<string, unknown>>;
  };
}

function createSession(): TraeACPAgentSession {
  const session = new TraeACPAgentSession(
    { provider: "acp", cwd: "/tmp/paseo-trae-acp-test" },
    {
      provider: "acp",
      logger: createTestLogger(),
      defaultCommand: ["traecli", "acp", "serve"],
      defaultModes: [],
      capabilities: DEFAULT_ACP_CAPABILITIES,
    },
  );
  asInternals<TraeSessionInternals>(session).sessionId = ROOT_THREAD_ID;
  return session;
}

function createSteeringSession() {
  const session = createSession();
  const internals = asInternals<TraeSessionInternals>(session);
  const prompt = vi.fn(() => new Promise<PromptResponse>(() => undefined));
  const extMethod = vi.fn(async () => ({ outcome: "injected" }));
  const cancel = vi.fn(async () => undefined);
  internals.connection = { prompt, extMethod, cancel };
  return { session, extMethod, cancel };
}

test("rejects steering for a different active turn", async () => {
  const { session, extMethod } = createSteeringSession();
  await session.startTurn("parent task");

  await expect(
    session.steerActiveTurn("follow up", { expectedTurnId: "another-turn" }),
  ).resolves.toEqual({ status: "unavailable" });
  expect(extMethod).not.toHaveBeenCalled();
});

test("steers the active Trae turn without canceling it", async () => {
  const { session, extMethod, cancel } = createSteeringSession();
  const { turnId } = await session.startTurn("parent task");

  await expect(
    session.steerActiveTurn(
      [
        { type: "text", text: "follow up" },
        { type: "image", data: "AA==", mimeType: "image/png" },
      ],
      { expectedTurnId: turnId, clientMessageId: "steer-message" },
    ),
  ).resolves.toEqual({ status: "accepted" });
  expect(extMethod).toHaveBeenCalledWith("_session/steering", {
    sessionId: ROOT_THREAD_ID,
    prompt: [
      { type: "text", text: "follow up" },
      { type: "image", data: "AA==", mimeType: "image/png" },
    ],
    _meta: { clientMessageId: "steer-message" },
  });
  expect(cancel).not.toHaveBeenCalled();
});

test.each([
  new RequestError(-32601, "method not found"),
  RequestError.invalidRequest(undefined, "no active turn to steer"),
  new RequestError(-32600, "no active turn to steer"),
])("falls back only when Trae definitively rejects steering", async (error) => {
  const { session, extMethod } = createSteeringSession();
  extMethod.mockRejectedValue(error);
  const { turnId } = await session.startTurn("parent task");

  await expect(session.steerActiveTurn("follow up", { expectedTurnId: turnId })).resolves.toEqual({
    status: "unavailable",
  });
});

test("does not treat an ambiguous steering error as unavailable", async () => {
  const { session, extMethod } = createSteeringSession();
  const error = RequestError.invalidRequest(undefined, "wrapped: no active turn to steer");
  extMethod.mockRejectedValue(error);
  const { turnId } = await session.startTurn("parent task");

  await expect(session.steerActiveTurn("follow up", { expectedTurnId: turnId })).rejects.toBe(
    error,
  );
});

test("clears only permissions pending when steering starts", async () => {
  const { session, extMethod, cancel } = createSteeringSession();
  const { turnId } = await session.startTurn("parent task");
  let finishSteering: (() => void) | undefined;
  extMethod.mockImplementation(
    () =>
      new Promise((resolve) => {
        finishSteering = () => resolve({ outcome: "injected" });
      }),
  );
  const request = (toolCallId: string) =>
    session.requestPermission({
      sessionId: ROOT_THREAD_ID,
      toolCall: {
        toolCallId,
        title: "Edit file",
        kind: "edit",
        status: "pending",
      },
      options: [
        { optionId: "allow-once", name: "Allow", kind: "allow_once" },
        { optionId: "reject-once", name: "Reject", kind: "reject_once" },
      ],
    } satisfies RequestPermissionRequest);
  const oldPermission = request("tool-before-steer");
  const oldRequestId = session.getPendingPermissions()[0]!.id;

  const steering = session.steerActiveTurn("do this instead", {
    expectedTurnId: turnId,
    clearPendingPermissions: true,
  });
  const newPermission = request("tool-after-steer");
  const newRequestId = session.getPendingPermissions().find(({ id }) => id !== oldRequestId)!.id;
  finishSteering?.();

  await expect(steering).resolves.toEqual({ status: "accepted" });
  await expect(oldPermission).resolves.toEqual({
    outcome: { outcome: "selected", optionId: "reject-once" },
  });
  expect(session.getPendingPermissions().map(({ id }) => id)).toEqual([newRequestId]);
  await session.respondToPermission(newRequestId, { behavior: "deny" });
  await expect(newPermission).resolves.toBeDefined();
  expect(cancel).not.toHaveBeenCalled();
});
