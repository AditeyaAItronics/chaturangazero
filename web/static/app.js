/**
 * ChaturangaZero & Connect-4 Interactive Arena Client
 */

// Piece Unicode maps
const CHATURANGA_PIECES = {
  1: { symbol: '♙', name: 'Padati (Pawn)', class: 'white-piece' },
  2: { symbol: '♘', name: 'Ashva (Knight)', class: 'white-piece' },
  3: { symbol: '♗', name: 'Gaja (Elephant)', class: 'white-piece' },
  4: { symbol: '♖', name: 'Ratha (Rook)', class: 'white-piece' },
  5: { symbol: '♕', name: 'Mantri (Counselor)', class: 'white-piece' },
  6: { symbol: '♔', name: 'Raja (King)', class: 'white-piece' },
  '-1': { symbol: '♟', name: 'Padati (Pawn)', class: 'black-piece' },
  '-2': { symbol: '♞', name: 'Ashva (Knight)', class: 'black-piece' },
  '-3': { symbol: '♝', name: 'Gaja (Elephant)', class: 'black-piece' },
  '-4': { symbol: '♜', name: 'Ratha (Rook)', class: 'black-piece' },
  '-5': { symbol: '♛', name: 'Mantri (Counselor)', class: 'black-piece' },
  '-6': { symbol: '♚', name: 'Raja (King)', class: 'black-piece' }
};

// Ashtapada special cross squares: a1, d1, e1, h1, a4, d4, e4, h4, a5, d5, e5, h5, a8, d8, e8, h8
const ASHTAPADA_SPECIAL_SQS = new Set([0, 3, 4, 7, 24, 27, 28, 31, 32, 35, 36, 39, 56, 59, 60, 63]);

class ArenaApp {
  constructor() {
    this.currentGame = 'chaturanga';
    this.gameState = null;
    this.models = { chaturanga: [], connect4: [] };
    this.selectedSq = null;
    this.legalTargets = new Map(); // to_sq -> action
    this.flipped = false;
    this.autoPlay = false;
    this.autoPlayDelay = 600; // ms delay between moves
    // Set when an AI move request fails: automatic AI moves stop (no endless
    // retry loop) until the next human action, new game, undo or Auto-Play toggle.
    // The "Trigger AI Move" button still works.
    this.autoAiPaused = false;
    this.autoPlayTimer = null;
    this.isThinking = false;
    this.isRequestPending = false;
    this.lastMove = null;

    this.initElements();
    this.bindEvents();
    this.init();
  }

  initElements() {
    // Header & Tabs
    this.tabChaturanga = document.getElementById('tab-chaturanga');
    this.tabConnect4 = document.getElementById('tab-connect4');
    this.btnNewGame = document.getElementById('btn-new-game');
    this.btnFlipBoard = document.getElementById('btn-flip-board');

    // Boards
    this.chaturangaContainer = document.getElementById('chaturanga-container');
    this.connect4Container = document.getElementById('connect4-container');
    this.chaturangaBoard = document.getElementById('chaturanga-board');
    this.connect4Board = document.getElementById('connect4-board');
    this.c4Pickers = document.getElementById('c4-pickers');
    this.coordsRanks = document.getElementById('coords-ranks');
    this.coordsFiles = document.getElementById('coords-files');

    // Player Status Bars
    this.p1Bar = document.getElementById('player-p1-bar');
    this.p2Bar = document.getElementById('player-p2-bar');
    this.p1DisplayName = document.getElementById('p1-display-name');
    this.p2DisplayName = document.getElementById('p2-display-name');
    this.p1ModelBadge = document.getElementById('p1-model-badge');
    this.p2ModelBadge = document.getElementById('p2-model-badge');
    this.p1CapturedBox = document.getElementById('p1-captured-box');
    this.p2CapturedBox = document.getElementById('p2-captured-box');
    this.p1MaterialScore = document.getElementById('p1-material-score');
    this.p2MaterialScore = document.getElementById('p2-material-score');
    this.p1Avatar = document.getElementById('p1-avatar');
    this.p2Avatar = document.getElementById('p2-avatar');
    this.p1ThinkingBadge = document.getElementById('p1-thinking-badge');
    this.p2ThinkingBadge = document.getElementById('p2-thinking-badge');

    // Floating Thinking Banner
    this.globalThinkingIndicator = document.getElementById('global-thinking-indicator');
    this.globalThinkingText = document.getElementById('global-thinking-text');

    // Selectors & Controls
    this.selectP1 = document.getElementById('select-p1');
    this.selectP2 = document.getElementById('select-p2');
    this.btnAiStep = document.getElementById('btn-ai-step');
    this.btnAutoPlay = document.getElementById('btn-autoplay');
    this.btnUndo = document.getElementById('btn-undo');

    // Speed Controls
    this.speedSlider = document.getElementById('speed-slider');
    this.speedDisplay = document.getElementById('speed-display');

    // AI Evaluation
    this.evalBarFill = document.getElementById('eval-bar-fill');
    this.evalValueText = document.getElementById('eval-value-text');
    this.candidatesList = document.getElementById('candidates-list');

    // Move History
    this.historyList = document.getElementById('history-list');
    this.plyCounter = document.getElementById('ply-counter');

    // Game Over Overlay
    this.gameOverOverlay = document.getElementById('game-over-overlay');
    this.gameOverTitle = document.getElementById('game-over-title');
    this.gameOverDesc = document.getElementById('game-over-desc');
    this.btnModalRestart = document.getElementById('btn-modal-restart');
  }

  bindEvents() {
    this.tabChaturanga.addEventListener('click', () => this.switchGame('chaturanga'));
    this.tabConnect4.addEventListener('click', () => this.switchGame('connect4'));
    this.btnNewGame.addEventListener('click', () => this.startNewGame());
    this.btnFlipBoard.addEventListener('click', () => this.toggleFlip());

    this.selectP1.addEventListener('change', () => this.startNewGame());
    this.selectP2.addEventListener('change', () => this.startNewGame());

    this.btnAiStep.addEventListener('click', () => this.stepAi());
    this.btnAutoPlay.addEventListener('click', () => this.toggleAutoPlay());
    this.btnUndo.addEventListener('click', () => this.undoMove());

    if (this.speedSlider) {
      this.speedSlider.addEventListener('input', (e) => {
        this.autoPlayDelay = parseInt(e.target.value, 10);
        if (this.speedDisplay) {
          this.speedDisplay.textContent = `${(this.autoPlayDelay / 1000).toFixed(2)}s`;
        }
      });
    }

    this.btnModalRestart.addEventListener('click', () => {
      this.gameOverOverlay.style.display = 'none';
      this.startNewGame();
    });
  }

  async init() {
    await this.fetchModels();
    await this.fetchCurrentState();
  }

  async fetchModels() {
    try {
      const res = await fetch('/api/models');
      this.models = await res.json();
      this.populateModelSelectors();
    } catch (err) {
      console.error('Failed to fetch models:', err);
    }
  }

  populateModelSelectors() {
    const list = this.models[this.currentGame] || [];
    this.selectP1.innerHTML = '';
    this.selectP2.innerHTML = '';

    list.forEach(m => {
      const opt1 = document.createElement('option');
      opt1.value = m.id;
      opt1.textContent = m.name;
      this.selectP1.appendChild(opt1);

      const opt2 = document.createElement('option');
      opt2.value = m.id;
      opt2.textContent = m.name;
      this.selectP2.appendChild(opt2);
    });

    // Default defaults: P1 Human, P2 best/latest AI
    if (this.currentGame === 'chaturanga') {
      this.selectP1.value = 'human';
      this.selectP2.value = 'az_latest';
    } else {
      this.selectP1.value = 'human';
      this.selectP2.value = 'az_final';
    }
  }

  async switchGame(game) {
    if (this.currentGame === game) return;
    if (this.autoPlayTimer) {
      clearTimeout(this.autoPlayTimer);
      this.autoPlayTimer = null;
    }
    this.currentGame = game;
    this.autoPlay = false;
    this.btnAutoPlay.textContent = '▶ Auto-Play: OFF';
    this.btnAutoPlay.classList.remove('btn-accent');
    this.btnAutoPlay.classList.add('btn-secondary');

    if (game === 'chaturanga') {
      this.tabChaturanga.classList.add('active');
      this.tabConnect4.classList.remove('active');
      this.chaturangaContainer.style.display = 'grid';
      this.connect4Container.style.display = 'none';
      this.p1Avatar.textContent = '♔';
      this.p2Avatar.textContent = '♚';
      this.p1MaterialScore.style.display = 'inline-block';
      this.p2MaterialScore.style.display = 'inline-block';
    } else {
      this.tabConnect4.classList.add('active');
      this.tabChaturanga.classList.remove('active');
      this.chaturangaContainer.style.display = 'none';
      this.connect4Container.style.display = 'flex';
      this.p1Avatar.textContent = '🔴';
      this.p2Avatar.textContent = '🟡';
      this.p1MaterialScore.style.display = 'none';
      this.p2MaterialScore.style.display = 'none';
      this.p1CapturedBox.innerHTML = '';
      this.p2CapturedBox.innerHTML = '';
    }

    this.populateModelSelectors();
    await this.startNewGame();
  }

  toggleFlip() {
    this.flipped = !this.flipped;
    this.render();
  }

  toggleAutoPlay() {
    this.autoPlay = !this.autoPlay;
    this.autoAiPaused = false;
    if (this.autoPlayTimer) {
      clearTimeout(this.autoPlayTimer);
      this.autoPlayTimer = null;
    }
    if (this.autoPlay) {
      this.btnAutoPlay.textContent = '⏸ Auto-Play: ON';
      this.btnAutoPlay.classList.remove('btn-secondary');
      this.btnAutoPlay.classList.add('btn-accent');
      this.checkAutoPlay();
    } else {
      this.btnAutoPlay.textContent = '▶ Auto-Play: OFF';
      this.btnAutoPlay.classList.remove('btn-accent');
      this.btnAutoPlay.classList.add('btn-secondary');
    }
  }

  // Should the AI move without being asked? Always when a human is playing
  // (the AI replies to the human's move, or opens when the human plays Black);
  // in AI-vs-AI games only while Auto-Play is on.
  aiMovesAutomatically() {
    const gs = this.gameState;
    if (!gs || this.autoAiPaused) return false;
    return this.autoPlay || gs.p1_type === 'human' || gs.p2_type === 'human';
  }

  checkAutoPlay() {
    if (this.autoPlayTimer) {
      clearTimeout(this.autoPlayTimer);
      this.autoPlayTimer = null;
    }
    if (!this.aiMovesAutomatically() || this.gameState.is_terminal || this.isRequestPending) {
      return;
    }
    if (this.gameState.current_player_type !== 'human') {
      this.autoPlayTimer = setTimeout(() => {
        this.autoPlayTimer = null;
        if (this.aiMovesAutomatically() && !this.gameState?.is_terminal && !this.isRequestPending) {
          this.stepAi();
        }
      }, this.autoPlayDelay);
    }
  }

  setThinkingState(isThinking) {
    this.isThinking = isThinking;

    // 1. Action button state
    if (this.btnAiStep) {
      this.btnAiStep.disabled = isThinking;
      if (isThinking) {
        this.btnAiStep.classList.add('is-thinking');
        this.btnAiStep.innerHTML = `
          <span class="thinking-spinner"></span>
          <span>Thinking (200 sims)...</span>
        `;
      } else {
        this.btnAiStep.classList.remove('is-thinking');
        this.btnAiStep.innerHTML = `⚡ Trigger AI Move`;
      }
    }

    // 2. Active Player Bar glow and badge
    const activePlayer = this.gameState ? this.gameState.side_to_move : 1;
    const activeBar = activePlayer === 1 ? this.p1Bar : this.p2Bar;
    const inactiveBar = activePlayer === 1 ? this.p2Bar : this.p1Bar;
    const activeBadge = activePlayer === 1 ? this.p1ThinkingBadge : this.p2ThinkingBadge;
    const inactiveBadge = activePlayer === 1 ? this.p2ThinkingBadge : this.p1ThinkingBadge;

    if (inactiveBar) inactiveBar.classList.remove('thinking');
    if (inactiveBadge) inactiveBadge.style.display = 'none';

    if (isThinking && activeBar) {
      activeBar.classList.add('thinking');
      if (activeBadge) activeBadge.style.display = 'inline-flex';
    } else if (activeBar) {
      activeBar.classList.remove('thinking');
      if (activeBadge) activeBadge.style.display = 'none';
    }

    // 3. Floating on-board banner
    if (this.globalThinkingIndicator) {
      if (isThinking) {
        const sideName = activePlayer === 1 ? (this.currentGame === 'chaturanga' ? 'White' : 'Red') : (this.currentGame === 'chaturanga' ? 'Black' : 'Yellow');
        const pType = activePlayer === 1 ? this.gameState?.p1_type : this.gameState?.p2_type;
        const pName = this.formatPlayerName(pType || 'az_latest');
        if (this.globalThinkingText) {
          this.globalThinkingText.textContent = `${sideName} (${pName}) is thinking (200 sims)...`;
        }
        this.globalThinkingIndicator.style.display = 'flex';
      } else {
        this.globalThinkingIndicator.style.display = 'none';
      }
    }

    // 4. Candidate List Thinking Animation
    if (isThinking && this.candidatesList) {
      this.candidatesList.innerHTML = `
        <div class="thinking-state-card">
          <div class="thinking-pulse-loader">
            <span class="dot"></span>
            <span class="dot"></span>
            <span class="dot"></span>
          </div>
          <div class="thinking-title">Evaluating Move Candidates...</div>
          <div class="thinking-subtitle">Running 200 MCTS rollouts & neural net value inference</div>
        </div>
      `;
    }
  }

  async startNewGame() {
    if (this.isRequestPending) return;
    this.isRequestPending = true;
    if (this.autoPlayTimer) {
      clearTimeout(this.autoPlayTimer);
      this.autoPlayTimer = null;
    }
    this.selectedSq = null;
    this.legalTargets.clear();
    this.lastMove = null;
    this.autoAiPaused = false;
    this.gameOverOverlay.style.display = 'none';

    try {
      const res = await fetch('/api/new_game', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          game: this.currentGame,
          p1_type: this.selectP1.value,
          p2_type: this.selectP2.value
        })
      });
      this.gameState = await res.json();
      this.render();
    } catch (err) {
      console.error('Failed to start new game:', err);
    } finally {
      this.isRequestPending = false;
      this.checkAutoPlay();
    }
  }

  async fetchCurrentState() {
    try {
      const res = await fetch('/api/state');
      this.gameState = await res.json();
      this.currentGame = this.gameState.game;

      if (this.currentGame === 'chaturanga') {
        this.tabChaturanga.classList.add('active');
        this.tabConnect4.classList.remove('active');
        this.chaturangaContainer.style.display = 'grid';
        this.connect4Container.style.display = 'none';
      } else {
        this.tabConnect4.classList.add('active');
        this.tabChaturanga.classList.remove('active');
        this.chaturangaContainer.style.display = 'none';
        this.connect4Container.style.display = 'flex';
      }

      this.populateModelSelectors();
      this.selectP1.value = this.gameState.p1_type;
      this.selectP2.value = this.gameState.p2_type;

      this.render();
      this.checkAutoPlay();
    } catch (err) {
      console.error('Failed to fetch state:', err);
    }
  }

  async makeMove(action) {
    if (this.isRequestPending || !this.gameState || this.gameState.is_terminal) return;
    this.isRequestPending = true;

    try {
      const res = await fetch('/api/make_move', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action })
      });
      if (res.ok) {
        this.gameState = await res.json();
        this.selectedSq = null;
        this.legalTargets.clear();
        this.autoAiPaused = false;
        this.render();
      } else {
        const errData = await res.json();
        console.warn('Move rejected:', errData);
      }
    } catch (err) {
      console.error('Failed to execute move:', err);
    } finally {
      this.isRequestPending = false;
      this.checkAutoPlay();
    }
  }

  async stepAi() {
    if (this.isRequestPending || !this.gameState || this.gameState.is_terminal) return;
    this.isRequestPending = true;
    this.setThinkingState(true);

    try {
      const res = await fetch('/api/ai_move', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
      });
      if (res.ok) {
        const data = await res.json();
        this.gameState = data;
        this.selectedSq = null;
        this.legalTargets.clear();

        if (data.played_action !== undefined) {
          this.lastMove = data.played_action;
        }

        if (data.ai_stats) {
          this.updateAiInsights(data.ai_stats);
        }

        this.render();
      } else {
        const errData = await res.json().catch(() => ({}));
        console.warn('AI Move failed; automatic AI moves paused:', errData);
        this.autoAiPaused = true;
      }
    } catch (err) {
      console.error('Error during AI move; automatic AI moves paused:', err);
      this.autoAiPaused = true;
    } finally {
      this.isRequestPending = false;
      this.setThinkingState(false);
      this.checkAutoPlay();
    }
  }

  async undoMove() {
    if (this.isRequestPending) return;
    this.isRequestPending = true;
    if (this.autoPlayTimer) {
      clearTimeout(this.autoPlayTimer);
      this.autoPlayTimer = null;
    }
    try {
      const res = await fetch('/api/undo', { method: 'POST' });
      if (res.ok) {
        this.autoAiPaused = false;
        this.gameState = await res.json();
        this.selectedSq = null;
        this.legalTargets.clear();
        this.render();
      }
    } catch (err) {
      console.error('Failed to undo move:', err);
    } finally {
      this.isRequestPending = false;
      this.checkAutoPlay();
    }
  }

  updateAiInsights(stats) {
    if (!stats) return;

    // Value evaluation bar
    if (stats.value !== undefined) {
      const val = stats.value; // -1.0 (Black/P2) to +1.0 (White/P1)
      const pct = Math.max(0, Math.min(100, ((val + 1) / 2) * 100));
      this.evalBarFill.style.width = `${pct}%`;
      const sign = val > 0 ? '+' : '';
      this.evalValueText.textContent = `${sign}${val.toFixed(2)}`;
    }

    // Candidate branch moves
    if (stats.top_moves && stats.top_moves.length > 0) {
      this.candidatesList.innerHTML = '';
      stats.top_moves.forEach((m, idx) => {
        const item = document.createElement('div');
        item.className = 'candidate-item';
        item.innerHTML = `
          <div class="candidate-item-left">
            <span class="candidate-rank">#${idx + 1}</span>
            <span class="candidate-move-name">${m.name}</span>
          </div>
          <div class="candidate-item-right">
            <span class="candidate-visits-badge">${m.visits} visits</span>
            <span class="candidate-pct">${m.pct}%</span>
          </div>
        `;
        this.candidatesList.appendChild(item);
      });
    }
  }

  render() {
    if (!this.gameState) return;

    this.renderPlayerBars();
    this.renderHistory();

    if (this.currentGame === 'chaturanga') {
      this.renderChaturanga();
    } else {
      this.renderConnect4();
    }

    this.checkGameOver();
  }

  renderPlayerBars() {
    const isP1Turn = this.gameState.side_to_move === 1;

    // Active Turn Highlights
    if (isP1Turn) {
      this.p1Bar.classList.add('active-turn');
      this.p2Bar.classList.remove('active-turn');
    } else {
      this.p2Bar.classList.add('active-turn');
      this.p1Bar.classList.remove('active-turn');
    }

    // Display Names & Badges
    const p1Label = this.currentGame === 'chaturanga' ? 'White' : 'Red';
    const p2Label = this.currentGame === 'chaturanga' ? 'Black' : 'Yellow';

    this.p1DisplayName.textContent = `${p1Label} (${this.formatPlayerName(this.gameState.p1_type)})`;
    this.p2DisplayName.textContent = `${p2Label} (${this.formatPlayerName(this.gameState.p2_type)})`;

    this.p1ModelBadge.textContent = `Type: ${this.gameState.p1_type}`;
    this.p2ModelBadge.textContent = `Type: ${this.gameState.p2_type}`;

    // Material Scores & Captured Pieces for Chaturanga
    if (this.currentGame === 'chaturanga') {
      this.p1MaterialScore.textContent = `Score: ${this.gameState.white_material.toFixed(1)}`;
      this.p2MaterialScore.textContent = `Score: ${this.gameState.black_material.toFixed(1)}`;

      this.p1CapturedBox.innerHTML = (this.gameState.white_captured || [])
        .map(p => CHATURANGA_PIECES[-p]?.symbol || '')
        .join(' ');

      this.p2CapturedBox.innerHTML = (this.gameState.black_captured || [])
        .map(p => CHATURANGA_PIECES[p]?.symbol || '')
        .join(' ');
    }
  }

  formatPlayerName(id) {
    if (id === 'human') return 'Human';
    if (id.startsWith('az_')) return 'AlphaZero';
    if (id === 'baseline_alphabeta') return 'Alpha-Beta (3-ply)';
    if (id === 'baseline_greedy') return 'Greedy Material';
    if (id === 'random') return 'Random';
    return id;
  }

  renderChaturanga() {
    // Render Coordinates
    const files = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'];
    const ranks = ['1', '2', '3', '4', '5', '6', '7', '8'];

    const displayRanks = this.flipped ? [...ranks] : [...ranks].reverse();
    const displayFiles = this.flipped ? [...files].reverse() : [...files];

    this.coordsRanks.innerHTML = displayRanks.map(r => `<div>${r}</div>`).join('');
    this.coordsFiles.innerHTML = displayFiles.map(f => `<div>${f}</div>`).join('');

    // Render Grid
    this.chaturangaBoard.innerHTML = '';
    const board = this.gameState.board; // 8x8 array [rank][file]

    const rankIndices = this.flipped ? [0, 1, 2, 3, 4, 5, 6, 7] : [7, 6, 5, 4, 3, 2, 1, 0];
    const fileIndices = this.flipped ? [7, 6, 5, 4, 3, 2, 1, 0] : [0, 1, 2, 3, 4, 5, 6, 7];

    rankIndices.forEach(r => {
      fileIndices.forEach(f => {
        const sqIdx = r * 8 + f;
        const piece = board[r][f];

        const sqDiv = document.createElement('div');
        sqDiv.className = `sq ${(r + f) % 2 === 0 ? 'dark' : 'light'}`;
        sqDiv.dataset.sq = sqIdx;
        sqDiv.dataset.rank = r;
        sqDiv.dataset.file = f;

        // Ashtapada special cross markings
        if (ASHTAPADA_SPECIAL_SQS.has(sqIdx)) {
          sqDiv.classList.add('ashtapada-special');
        }

        // Selection highlight
        if (this.selectedSq === sqIdx) {
          sqDiv.classList.add('selected');
        }

        // Legal target highlight
        if (this.legalTargets.has(sqIdx)) {
          if (piece !== 0) {
            sqDiv.classList.add('legal-capture');
          } else {
            sqDiv.classList.add('legal-target');
          }
        }

        // Piece rendering
        if (piece !== 0) {
          const pieceInfo = CHATURANGA_PIECES[piece];
          if (pieceInfo) {
            const pieceDiv = document.createElement('span');
            pieceDiv.className = `piece ${pieceInfo.class}`;
            pieceDiv.textContent = pieceInfo.symbol;
            pieceDiv.title = pieceInfo.name;
            sqDiv.appendChild(pieceDiv);
          }
        }

        // Click handler
        sqDiv.addEventListener('click', () => this.handleChaturangaSqClick(sqIdx, piece));
        this.chaturangaBoard.appendChild(sqDiv);
      });
    });
  }

  handleChaturangaSqClick(sqIdx, piece) {
    if (this.gameState.is_terminal || this.isRequestPending) return;

    // If human clicked a legal destination for selected piece
    if (this.legalTargets.has(sqIdx)) {
      const action = this.legalTargets.get(sqIdx);
      this.makeMove(action);
      return;
    }

    // Only allow human selection on human's turn
    if (this.gameState.current_player_type !== 'human') {
      return;
    }

    // Check if clicked square has a piece belonging to side to move
    const side = this.gameState.side_to_move;
    const isFriendlyPiece = (side === 1 && piece > 0) || (side === -1 && piece < 0);

    if (isFriendlyPiece) {
      if (this.selectedSq === sqIdx) {
        this.selectedSq = null;
        this.legalTargets.clear();
      } else {
        this.selectedSq = sqIdx;
        this.legalTargets.clear();

        // Filter legal moves originating from this square
        (this.gameState.legal_moves || []).forEach(m => {
          if (m.from_sq === sqIdx) {
            this.legalTargets.set(m.to_sq, m.action);
          }
        });
      }
      this.render();
    } else {
      this.selectedSq = null;
      this.legalTargets.clear();
      this.render();
    }
  }

  renderConnect4() {
    // Top Column Drop Pickers
    this.c4Pickers.innerHTML = '';
    const legalCols = new Set((this.gameState.legal_moves || []).map(m => m.column));
    const isHumanTurn = this.gameState.current_player_type === 'human' && !this.gameState.is_terminal;

    for (let c = 0; c < 7; c++) {
      const btn = document.createElement('button');
      btn.className = 'c4-col-drop-btn';
      btn.textContent = '▼';
      btn.title = `Drop in Column ${c + 1}`;
      btn.disabled = !legalCols.has(c) || !isHumanTurn || this.isRequestPending;

      btn.addEventListener('click', () => {
        if (legalCols.has(c) && !this.isRequestPending) {
          this.makeMove(c);
        }
      });
      this.c4Pickers.appendChild(btn);
    }

    // Connect-4 Board: 6 rows (r=5 at top to r=0 at bottom), 7 cols
    this.connect4Board.innerHTML = '';
    const board = this.gameState.board; // [row][col]

    for (let r = 5; r >= 0; r--) {
      for (let c = 0; c < 7; c++) {
        const val = board[r][c];
        const cell = document.createElement('div');
        cell.className = 'c4-cell';

        if (val === 1) {
          cell.classList.add('p1-token');
        } else if (val === -1) {
          cell.classList.add('p2-token');
        }

        cell.addEventListener('click', () => {
          if (isHumanTurn && legalCols.has(c) && !this.isRequestPending) {
            this.makeMove(c);
          }
        });

        this.connect4Board.appendChild(cell);
      }
    }
  }

  renderHistory() {
    const moves = this.gameState.move_history || [];
    this.plyCounter.textContent = `Ply: ${moves.length}`;
    this.historyList.innerHTML = '';

    for (let i = 0; i < moves.length; i += 2) {
      const moveNum = Math.floor(i / 2) + 1;
      const m1 = moves[i];
      const m2 = moves[i + 1] || '';

      const row = document.createElement('div');
      row.className = 'history-row';
      row.innerHTML = `
        <span class="history-num">${moveNum}.</span>
        <span class="history-move">${m1}</span>
        <span class="history-move">${m2}</span>
      `;
      this.historyList.appendChild(row);
    }

    // Auto-scroll history
    this.historyList.scrollTop = this.historyList.scrollHeight;
  }

  checkGameOver() {
    if (!this.gameState || !this.gameState.is_terminal) {
      this.gameOverOverlay.style.display = 'none';
      return;
    }

    if (this.autoPlayTimer) {
      clearTimeout(this.autoPlayTimer);
      this.autoPlayTimer = null;
    }
    this.setThinkingState(false);

    const z = this.gameState.result; // from side-to-move's perspective
    const side = this.gameState.side_to_move;
    const reason = this.gameState.terminal_reason || '';

    let winnerText = '';
    let icon = '🏆';

    if (z === 0) {
      winnerText = 'Draw / Stalemate';
      icon = '🤝';
    } else {
      // The player whose turn it is got score z
      // If z = -1, the opponent won
      const winningSide = z > 0 ? side : -side;
      if (this.currentGame === 'chaturanga') {
        winnerText = winningSide === 1 ? 'White Wins!' : 'Black Wins!';
      } else {
        winnerText = winningSide === 1 ? 'Red Wins!' : 'Yellow Wins!';
      }
    }

    this.gameOverTitle.textContent = winnerText;
    this.gameOverDesc.textContent = `Game concluded. Reason: ${this.formatTerminalReason(reason)}`;
    this.gameOverOverlay.style.display = 'flex';
  }

  formatTerminalReason(reason) {
    const map = {
      'raja_captured': 'Raja captured (direct victory)',
      'bare_king': 'Bare King (all opposing army captured)',
      'king_vs_king': 'King vs King (insufficient material draw)',
      'move_cap': 'Move cap limit reached (Draw)',
      'stalemate': 'Stalemate (win for the stalemated player per authentic rules)',
      'four_in_a_row': 'Four connected tokens in a row',
      'board_full': 'Grid completely filled (Draw)'
    };
    return map[reason] || reason || 'Terminal state reached';
  }
}

// Bootstrap once DOM ready
document.addEventListener('DOMContentLoaded', () => {
  window.arenaApp = new ArenaApp();
});
