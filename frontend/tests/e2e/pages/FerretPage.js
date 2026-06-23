/**
 * Page Object Model for the Ferret chat UI.
 *
 * All selectors reference the semantic IDs already present in index.html.
 * No data-testid attributes are added to the app markup.
 */
export class FerretPage {
  constructor(page) {
    this.page = page;

    // Mode picker
    this.pickAsk   = page.locator('#pick-ask');
    this.pickDeep  = page.locator('#pick-deep');
    this.modePicker = page.locator('#mode-picker');

    // Chat controls
    this.chatInput = page.locator('#chat-input');
    this.chatBtn   = page.locator('#chat-btn');
    this.chatList  = page.locator('#chat-list');

    // arXiv / Deep-dive
    this.arxivInput      = page.locator('#arxiv-input');
    this.arxivBtn        = page.locator('#arxiv-btn');
    this.ingestionStatus = page.locator('#ingestion-status');

    // Citations tray (rendered inside .chat-list, first .citations div)
    this.citations = page.locator('.citations').first();

    // Session sidebar
    this.sessionList = page.locator('#session-list');
    this.newChatBtn  = page.locator('#new-chat-btn');
  }

  async goto() {
    await this.page.goto('/');
  }

  /** Wait for the mode picker to be visible (initial empty state). */
  async waitForModePicker() {
    await this.modePicker.waitFor({ state: 'visible' });
  }

  /** Click Ask and wait for the chat input to become enabled. */
  async selectAskMode() {
    await this.pickAsk.click();
    await this.chatInput.waitFor({ state: 'visible' });
    // enableChat removes the disabled attribute
    await this.page.waitForFunction(
      () => !document.getElementById('chat-input').disabled,
      { timeout: 8000 }
    );
  }

  /** Type a message and click Send; returns immediately after click. */
  async sendMessage(text) {
    await this.chatInput.fill(text);
    await this.chatBtn.click();
  }

  /** Return the text content of all .msg elements in the chat list. */
  async getMessages() {
    return this.chatList.locator('.msg').allTextContents();
  }

  /** Return the last assistant message element. */
  lastAssistantMessage() {
    return this.chatList.locator('.msg-assistant').last();
  }

  /** Return the first citation pill rendered after a response. */
  firstCitationPill() {
    return this.citations.locator('.citation-pill').first();
  }
}
