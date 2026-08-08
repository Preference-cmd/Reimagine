import { WelcomeScreen } from "./WelcomeScreen";

/**
 * ChatView — main chat/conversation view.
 * Shows WelcomeScreen when no conversation is active.
 * Will be extended with message list + input when chat state is wired.
 */
export function ChatView() {
  return <WelcomeScreen />;
}
