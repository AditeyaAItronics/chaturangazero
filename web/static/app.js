/**
 * ChaturangaZero & Connect-4 Interactive Arena Client
 */

// Piece Unicode maps
// Piece drawings live in pieces.js (pieceSVG, PIECE_NAMES_UI)
const CHATURANGA_PIECES = Object.fromEntries(
  [1, 2, 3, 4, 5, 6, -1, -2, -3, -4, -5, -6].map(code => [code, {
    svg: pieceSVG(code),
    name: PIECE_NAMES_UI[Math.abs(code)],
    class: code > 0 ? 'white-piece' : 'black-piece',
  }])
);

// Move-quality labels from /api/analyze_move (symbol carries the meaning; colour helps)
const MOVE_LABEL_CLASS = { '??': 'lab-blunder', '?': 'lab-mistake', '?!': 'lab-inaccuracy',
                           '!': 'lab-good', '!!': 'lab-brilliant' };
const MOVE_LABEL_NAME = { '??': 'Blunders', '?': 'Mistakes', '?!': 'Inaccuracies',
                          '!': 'Good (only) moves', '!!': 'Brilliant moves' };

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
    this.autoPlay = true;     // AI vs AI plays on unless paused
    this.autoPlayDelay = 600; // ms delay between moves
    this.reviewIndex = null;  // ply shown while reviewing a finished game
    this.positions = [];      // every position of the game, from /api/positions
    this.analysis = [];       // per move: {label, reason, ...} from /api/analyze_move
    this.analysing = false;
    this.overlayDismissed = false; // game-over card closed for this game (review/analyse)
    this.playSims = 200;      // set by the search-strength slider
    // Set when an AI move request fails: automatic AI moves stop (no endless
    // retry loop) until the next human action, new game, undo, or Pause/Resume.
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
    // Initial avatars and the rules panel's piece key use the SVG piece set
    this.p1Avatar.innerHTML = pieceSVG(6);
    this.p2Avatar.innerHTML = pieceSVG(-6);
    document.querySelectorAll('.key-icon').forEach(el => {
      el.innerHTML = pieceSVG(Number(el.dataset.piece));
    });
    this.loadModelCard();
    this.p1ThinkingBadge = document.getElementById('p1-thinking-badge');
    this.p2ThinkingBadge = document.getElementById('p2-thinking-badge');

    // Floating Thinking Banner
    this.globalThinkingIndicator = document.getElementById('global-thinking-indicator');
    this.globalThinkingText = document.getElementById('global-thinking-text');

    // Selectors & Controls
    this.selectP1 = document.getElementById('select-p1');
    this.selectP2 = document.getElementById('select-p2');
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

    this.btnAutoPlay.addEventListener('click', () => this.toggleAutoPlay());
    this.btnUndo.addEventListener('click', () => this.undoMove());

    this.strengthSlider = document.getElementById('strength-slider');
    this.strengthSlider.addEventListener('input', (e) => {
      this.playSims = parseInt(e.target.value, 10);
      document.getElementById('strength-display').textContent = `${this.playSims} sims`;
      const cardSims = document.getElementById('model-sims');
      if (cardSims) cardSims.textContent = this.playSims;
    });

    document.getElementById('btn-analyse').addEventListener('click', () => this.analyseGame());
    document.getElementById('btn-modal-analyse').addEventListener('click', () => this.analyseGame());

    // Reviewing a finished game
    this.reviewNav = document.getElementById('review-nav');
    document.getElementById('btn-modal-review').addEventListener('click', () => this.startReview());
    document.getElementById('review-first').addEventListener('click', () => this.reviewGo(0));
    document.getElementById('review-prev').addEventListener('click', () => this.reviewStep(-1));
    document.getElementById('review-next').addEventListener('click', () => this.reviewStep(1));
    document.getElementById('review-last').addEventListener('click', () => this.reviewGo(Infinity));
    document.addEventListener('keydown', (e) => {
      const typing = e.target instanceof Element && e.target.closest('input, select, textarea');
      if (this.reviewIndex === null || typing) return;
      if (e.key === 'ArrowLeft') { this.reviewStep(-1); e.preventDefault(); }
      if (e.key === 'ArrowRight') { this.reviewStep(1); e.preventDefault(); }
    });

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
      this.selectP2.value = 'az_best';
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
    this.autoPlay = true;
    this.btnAutoPlay.textContent = '⏸ Pause AI vs AI';

    if (game === 'chaturanga') {
      this.tabChaturanga.classList.add('active');
      this.tabConnect4.classList.remove('active');
      this.chaturangaContainer.style.display = 'grid';
      this.connect4Container.style.display = 'none';
      document.getElementById('rules-panel').style.display = '';
      this.p1Avatar.innerHTML = pieceSVG(6);
      this.p2Avatar.innerHTML = pieceSVG(-6);
      this.p1MaterialScore.style.display = 'inline-block';
      this.p2MaterialScore.style.display = 'inline-block';
    } else {
      this.tabConnect4.classList.add('active');
      this.tabChaturanga.classList.remove('active');
      this.chaturangaContainer.style.display = 'none';
      this.connect4Container.style.display = 'flex';
      document.getElementById('rules-panel').style.display = 'none';
      this.p1Avatar.textContent = '🔴';
      this.p2Avatar.textContent = '🟡';
      this.p1MaterialScore.style.display = 'none';
      this.p2MaterialScore.style.display = 'none';
      this.p1CapturedBox.innerHTML = '';
      this.p2CapturedBox.innerHTML = '';
    }

    this.populateModelSelectors();
    this.updateEvalLabels();
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
      this.btnAutoPlay.textContent = '⏸ Pause AI vs AI';
      this.btnAutoPlay.classList.remove('btn-accent');
      this.checkAutoPlay();
    } else {
      this.btnAutoPlay.textContent = '▶ Resume AI vs AI';
      this.btnAutoPlay.classList.add('btn-accent');
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
          this.globalThinkingText.textContent = `${sideName} (${pName}) is thinking…`;
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
          <div class="thinking-title">Searching…</div>
          <div class="thinking-subtitle">${this.playSims} Monte Carlo tree search simulations, each guided by the neural network's move priors and position value</div>
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
    this.reviewIndex = null;
    this.positions = [];
    this.analysis = [];
    this.overlayDismissed = false;
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
      if (this.selectP1.value === 'human') this.flipped = false;
      else if (this.selectP2.value === 'human') this.flipped = true;
      // (AI vs AI keeps whatever Flip View says)
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
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ sims: this.playSims })
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
        this.reviewIndex = null;
        this.positions = [];
        this.analysis = [];
        this.overlayDismissed = false;
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

  async loadModelCard() {
    const badge = document.getElementById('model-iter-badge');
    const box = document.getElementById('model-stats');
    try {
      const res = await fetch('/api/model_info?id=az_best');
      const info = await res.json();
      if (!info.available) {
        badge.textContent = 'no model';
        box.innerHTML = '<div class="empty-state">No trained checkpoint found in runs/chaturanga.</div>';
        return;
      }
      badge.textContent = info.iteration ? `iteration ${info.iteration}` : 'best model';
      const fmt = n => (n === undefined || n === null) ? null : Number(n).toLocaleString();
      const rows = [
        ['Network', info.blocks ? `ResNet, ${info.blocks} blocks × ${info.channels} channels` : null],
        ['Parameters', fmt(info.parameters)],
        ['Training', info.iterations_trained ? `${fmt(info.iterations_trained)} iterations · ${fmt(info.selfplay_games)} self-play games` : null],
        ['Gradient steps', fmt(info.train_steps)],
        ['Promotions', info.arena_matches ? `${info.promotions} of ${info.arena_matches} arena matches won` : null],
        ['Compute', info.hours ? `${info.hours} h of self-play training` : null],
        ['Search here', `<span id="model-sims">${this.playSims}</span> simulations per move (slider)`],
      ].filter(([, v]) => v);
      box.innerHTML = rows.map(([k, v]) => `<div><dt>${k}</dt><dd>${v}</dd></div>`).join('');
    } catch (err) {
      badge.textContent = 'unavailable';
      console.error('Model info failed:', err);
    }
  }

  updateEvalLabels() {
    const c4 = this.currentGame === 'connect4';
    document.getElementById('label-p1').textContent = c4 ? 'Red' : 'White';
    document.getElementById('label-p2').textContent = c4 ? 'Yellow' : 'Black';
    document.getElementById('eval-label-p1').textContent = c4 ? 'Red' : 'White';
    document.getElementById('eval-label-p2').textContent = c4 ? 'Yellow' : 'Black';
    document.getElementById('model-card').style.display = c4 ? 'none' : '';
  }

  // Load every position of the finished game, then show ply `target`
  async startReview(target = Infinity) {
    if (this.currentGame !== 'chaturanga') return;
    try {
      const res = await fetch('/api/positions');
      this.positions = (await res.json()).positions || [];
    } catch (err) {
      console.error('Could not load the game for review:', err);
      return;
    }
    if (!this.positions.length) return;
    this.overlayDismissed = true;
    this.gameOverOverlay.style.display = 'none';
    this.reviewGo(target);
  }

  reviewGo(index) {
    if (!this.positions.length) { this.startReview(index); return; }
    this.reviewIndex = Math.max(0, Math.min(this.positions.length - 1, index));
    this.render();
  }

  async reviewStep(delta) {
    if (!this.positions.length) await this.startReview(Infinity);
    if (!this.positions.length) return;
    const from = this.reviewIndex === null ? this.positions.length - 1 : this.reviewIndex;
    this.reviewGo(from + delta);
  }

  updateReviewNav() {
    const show = this.currentGame === 'chaturanga' && this.gameState?.is_terminal;
    this.reviewNav.hidden = !show;
    if (!show) return;
    const last = Math.max(0, (this.positions.length || (this.gameState.ply_count + 1)) - 1);
    const at = this.reviewIndex === null ? last : this.reviewIndex;
    document.getElementById('review-pos').textContent =
      this.reviewIndex === null ? 'Review' : `${at} / ${last}`;
    document.getElementById('review-pos').title = 'Ply shown / last ply (← and → keys step)';
    document.getElementById('review-first').disabled = this.reviewIndex !== null && at === 0;
    document.getElementById('review-prev').disabled = this.reviewIndex !== null && at === 0;
    document.getElementById('review-next').disabled = this.reviewIndex === null || at === last;
    document.getElementById('review-last').disabled = this.reviewIndex === null || at === last;
  }

  // ------------------------------------------------------------- analysis
  // After a Chaturanga game, judge every move with the best model's search,
  // one move per request so the button can show progress.
  async analyseGame() {
    if (this.analysing || !this.gameState?.is_terminal) return;
    this.overlayDismissed = true;
    this.gameOverOverlay.style.display = 'none';
    const moves = (this.gameState.move_history || []).length;
    this.analysing = true;
    this.analysis = new Array(moves).fill(null);
    try {
      for (let ply = 0; ply < moves; ply++) {
        this.renderAnalysisControls(ply, moves);
        const res = await fetch('/api/analyze_move', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ ply })
        });
        if (!res.ok) throw new Error((await res.json().catch(() => ({}))).error || res.status);
        this.analysis[ply] = await res.json();
      }
    } catch (err) {
      console.error('Analysis failed:', err);
      this.analysis = [];
    } finally {
      this.analysing = false;
      this.render();
    }
  }

  renderAnalysisControls(done, total) {
    const btn = document.getElementById('btn-analyse');
    const box = document.getElementById('analysis-summary');
    const canAnalyse = this.currentGame === 'chaturanga' && this.gameState?.is_terminal
      && (this.gameState.move_history || []).length > 0;
    btn.hidden = !canAnalyse;
    if (this.analysing) {
      btn.disabled = true;
      btn.textContent = `Analysing ${done ?? 0}/${total ?? '…'}`;
    } else {
      btn.disabled = false;
      btn.textContent = this.analysis.length ? '🔍 Re-analyse' : '🔍 Analyse';
    }
    const complete = this.analysis.length && this.analysis.every(Boolean);
    box.hidden = !complete;
    if (!complete) return;
    const sides = ['White', 'Black'];
    const order = ['!!', '!', '?!', '?', '??'];
    box.innerHTML = sides.map(side => {
      const mine = this.analysis.filter(a => a.side === side && a.label);
      const chips = order.map(l => {
        const n = mine.filter(a => a.label === l).length;
        return n ? `<span class="move-label ${MOVE_LABEL_CLASS[l]}" title="${MOVE_LABEL_NAME[l]}">${n}× ${l}</span>` : '';
      }).join('');
      return `<div><strong>${side}</strong> ${chips || '<span class="none">no labels</span>'}</div>`;
    }).join('') + '<div class="judge">Judged by the best model\'s search (200 simulations per position)</div>';
  }

  // ------------------------------------------------------------- momentum
  // White's expected score (from each AlphaZero move's search) per ply. Above
  // the 50% line White is better (gold), below it Black (rose); position, not
  // only colour, carries the side. Hover for the value; after the game, click
  // to jump to that position.
  renderMomentum() {
    const card = document.getElementById('momentum-card');
    if (this.currentGame !== 'chaturanga') { card.style.display = 'none'; return; }
    card.style.display = '';
    const box = document.getElementById('momentum-chart');
    const evals = this.gameState.evals || [];
    const plies = Math.max(evals.length, 1);
    // Points: ply 0 starts level; then every ply with a known value. A finished
    // analysis judges every position with one model, so it replaces them.
    const analysed = this.analysis.length === evals.length && evals.length > 0
      && this.analysis.every(Boolean);
    const pts = analysed
      ? [[0, this.analysis[0].white_before], ...this.analysis.map((a, i) => [i + 1, a.white_after])]
      : [[0, 0.5]];
    if (!analysed) evals.forEach((v, i) => { if (v !== null && v !== undefined) pts.push([i + 1, v]); });

    const W = 600, H = 84, PAD_L = 4, PAD_R = 4, PAD_T = 6, PAD_B = 6;
    const x = ply => PAD_L + (ply / plies) * (W - PAD_L - PAD_R);
    const y = v => PAD_T + (1 - v) * (H - PAD_T - PAD_B);
    const mid = y(0.5);
    const line = pts.map(([p, v], i) => `${i ? 'L' : 'M'}${x(p).toFixed(1)},${y(v).toFixed(1)}`).join('');
    const area = `${line}L${x(pts[pts.length - 1][0]).toFixed(1)},${mid}L${x(0)},${mid}Z`;
    const marker = this.reviewIndex !== null
      ? `<line class="mom-cursor" x1="${x(this.reviewIndex)}" x2="${x(this.reviewIndex)}" y1="${PAD_T}" y2="${H - PAD_B}"/>`
      : '';
    const lastPt = pts[pts.length - 1];
    box.innerHTML = `
      <svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="none" class="mom-svg">
        <defs>
          <clipPath id="mom-above"><rect x="0" y="0" width="${W}" height="${mid}"/></clipPath>
          <clipPath id="mom-below"><rect x="0" y="${mid}" width="${W}" height="${H - mid}"/></clipPath>
        </defs>
        <path class="mom-area white-ahead" d="${area}" clip-path="url(#mom-above)"/>
        <path class="mom-area black-ahead" d="${area}" clip-path="url(#mom-below)"/>
        <line class="mom-mid" x1="0" x2="${W}" y1="${mid}" y2="${mid}"/>
        <path class="mom-line" d="${line}"/>
        ${marker}
        ${analysed ? this.analysis.map((a, i) => (a.label === '??' || a.label === '!!')
          ? `<circle class="mom-mark ${MOVE_LABEL_CLASS[a.label]}" cx="${x(i + 1)}" cy="${y(a.white_after)}" r="5"/>`
          : '').join('') : ''}
        <line class="mom-hover" id="mom-hover" x1="0" x2="0" y1="${PAD_T}" y2="${H - PAD_B}" visibility="hidden"/>
      </svg>
      <span class="mom-label top">White better</span>
      <span class="mom-label bottom">Black better</span>`;

    const pct = v => Math.round(v * 100);
    document.getElementById('momentum-now').textContent =
      pts.length > 1 ? `White ${pct(lastPt[1])}% – Black ${100 - pct(lastPt[1])}%` : 'No AI evaluation yet';
    box.setAttribute('aria-label', pts.length > 1
      ? `White's expected score over ${plies} plies; now ${pct(lastPt[1])}%`
      : "White's expected score: no AI evaluation yet");

    // Hover: nearest evaluated ply; click: review that position after the game
    const tip = document.getElementById('momentum-tip');
    const hover = box.querySelector('#mom-hover');
    const nearest = ev => {
      const r = box.getBoundingClientRect();
      const ply = Math.round(((ev.clientX - r.left) / r.width) * plies);
      return pts.reduce((best, p) => Math.abs(p[0] - ply) < Math.abs(best[0] - ply) ? p : best, pts[0]);
    };
    box.onmousemove = ev => {
      const [p, v] = nearest(ev);
      hover.setAttribute('x1', x(p)); hover.setAttribute('x2', x(p));
      hover.setAttribute('visibility', 'visible');
      const move = p === 0 ? 'Start' : `After ${this.gameState.move_history?.[p - 1] || 'ply ' + p}`;
      const lab = p > 0 && this.analysis[p - 1]?.label ? ` · ${this.analysis[p - 1].label}` : '';
      tip.textContent = `${move}${lab} · White ${pct(v)}% – Black ${100 - pct(v)}%`;
      tip.hidden = false;
      const r = box.getBoundingClientRect();
      tip.style.left = `${Math.min(Math.max((x(p) / W) * r.width, 70), r.width - 70)}px`;
    };
    box.onmouseleave = () => { tip.hidden = true; hover.setAttribute('visibility', 'hidden'); };
    box.onclick = ev => {
      if (!this.gameState.is_terminal) return;
      this.reviewGo(nearest(ev)[0]);
    };
    box.classList.toggle('clickable', !!this.gameState.is_terminal);
    box.title = this.gameState.is_terminal ? 'Click to review the game from that point' : '';
  }

  updateAiInsights(stats) {
    if (!stats) return;

    // Value evaluation bar
    if (stats.value !== undefined) {
      // Server sends the search value from White's / Red's view (-1 .. +1);
      // shown as expected score, where a draw counts as half
      const val = stats.value;
      const pct = Math.max(0, Math.min(100, ((val + 1) / 2) * 100));
      this.evalBarFill.style.width = `${pct}%`;
      this.evalValueText.textContent = `${Math.round(pct)}% – ${100 - Math.round(pct)}%`;
    }

    // The network's instinct (raw policy) against what the search chose
    const instinctLine = document.getElementById('instinct-line');
    if (stats.instinct && stats.top_moves && stats.top_moves.length) {
      const chosen = stats.top_moves[0];
      instinctLine.hidden = false;
      instinctLine.className = `instinct-line ${stats.search_agrees ? 'agrees' : 'overruled'}`;
      instinctLine.innerHTML = stats.search_agrees
        ? `Search <strong>confirmed</strong> the network's first instinct: ${stats.instinct.name} (${Math.round(stats.instinct.prior)}% → ${Math.round(chosen.pct)}%).`
        : `Search <strong>overruled</strong> the network: its instinct was ${stats.instinct.name} (${Math.round(stats.instinct.prior)}%), but ${stats.sims} simulations chose ${chosen.name}.`;
    } else {
      instinctLine.hidden = true;
    }

    // Candidate branch moves
    if (stats.top_moves && stats.top_moves.length > 0) {
      this.candidatesList.innerHTML = '';
      stats.top_moves.forEach((m, idx) => {
        const item = document.createElement('div');
        item.className = 'candidate-item';
        const icon = m.piece ? `<span class="candidate-icon">${pieceSVG(m.piece)}</span>` : '';
        const total = stats.total_visits || stats.sims;
        item.title = (m.prior !== undefined ? `Network instinct: ${m.prior}% · ` : '')
          + `after search: ${m.visits} of ${total} simulations (${m.pct}%)`;
        item.innerHTML = `
          <div class="candidate-item-left">
            <span class="candidate-rank">${idx === 0 ? 'Played' : '#' + (idx + 1)}</span>
            ${icon}
            <span class="candidate-move-name">${m.name}</span>
          </div>
          <div class="candidate-item-right">
            <span class="candidate-visits-badge" title="${m.visits} of ${total} search visits">${m.visits}/${total}</span>
            <span class="candidate-pct">${Math.round(m.pct)}%</span>
            ${m.prior !== undefined ? `<span class="candidate-prior" style="--share:${m.prior}%"></span>` : ''}
            <span class="candidate-share" style="--share:${m.pct}%"></span>
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
    this.updateReviewNav();
    this.renderAnalysisControls();
    this.renderMomentum();

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

    this.p1ModelBadge.textContent = this.playerDescription(this.gameState.p1_type);
    this.p2ModelBadge.textContent = this.playerDescription(this.gameState.p2_type);

    // Material Scores & Captured Pieces for Chaturanga
    if (this.currentGame === 'chaturanga') {
      this.p1MaterialScore.textContent = `Material ${this.gameState.white_material.toFixed(1)}`;
      this.p2MaterialScore.textContent = `Material ${this.gameState.black_material.toFixed(1)}`;
      this.p1MaterialScore.title = this.p2MaterialScore.title =
        'Piece values: Ratha 5, Ashva 3, Gaja 1.5, Mantri 1.5, Padati 1 (Raja not counted)';

      this.p1CapturedBox.innerHTML = (this.gameState.white_captured || [])
        .map(p => CHATURANGA_PIECES[-p]?.svg || '')
        .join(' ');

      this.p2CapturedBox.innerHTML = (this.gameState.black_captured || [])
        .map(p => CHATURANGA_PIECES[p]?.svg || '')
        .join(' ');
    }
  }

  playerDescription(id) {
    if (id === 'human') return 'Human player';
    const opt = [...this.selectP1.options].find(o => o.value === id);
    return opt ? opt.textContent : id;
  }

  formatPlayerName(id) {
    if (id === 'human') return 'Human';
    if (id.startsWith('az_')) return 'AlphaZero';
    if (id.startsWith('ckpt_')) return `AlphaZero ${Number(id.slice(5)) + 1}`;
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

    // Render Grid. While reviewing, show that ply's position (no moves allowed)
    const reviewed = this.reviewIndex !== null ? this.positions[this.reviewIndex] : null;
    const gs = reviewed ? { ...this.gameState, ...reviewed, legal_moves: [] } : this.gameState;
    this.chaturangaBoard.innerHTML = '';
    const board = gs.board; // 8x8 array [rank][file]

    const rankIndices = this.flipped ? [0, 1, 2, 3, 4, 5, 6, 7] : [7, 6, 5, 4, 3, 2, 1, 0];
    const enemyRaja = -6 * gs.side_to_move;
    const capturableRaja = new Set((gs.legal_moves || [])
      .map(m => m.to_sq)
      .filter(t => board[Math.floor(t / 8)][t % 8] === enemyRaja));
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

        // Last move played
        const last = gs.last_move;
        if (last && sqIdx === last.from_sq) sqDiv.classList.add('last-move-from');
        if (last && sqIdx === last.to_sq) sqDiv.classList.add('last-move-to');

        // A Raja the side to move can capture right now (there is no check:
        // the warning is all you get)
        if (capturableRaja.has(sqIdx)) {
          sqDiv.classList.add('raja-capturable');
          sqDiv.title = 'This Raja can be captured this move';
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
            pieceDiv.innerHTML = pieceInfo.svg;
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
      const cls = ply => `history-move${this.reviewIndex === ply ? ' reviewing' : ''}`;
      const mark = idx => {
        const a = this.analysis[idx];
        if (!a || !a.label) return '';
        const extra = a.better ? ` Better: ${a.better}.` : '';
        return `<span class="move-label ${MOVE_LABEL_CLASS[a.label]}" title="${a.reason}.${extra}">${a.label}</span>`;
      };
      row.innerHTML = `
        <span class="history-num">${moveNum}.</span>
        <span class="${cls(i + 1)}" data-ply="${i + 1}">${m1}${mark(i)}</span>
        <span class="${cls(i + 2)}" data-ply="${i + 2}">${m2}${m2 ? mark(i + 1) : ''}</span>
      `;
      if (this.gameState.is_terminal && this.currentGame === 'chaturanga') {
        row.querySelectorAll('.history-move').forEach(el => {
          if (!el.textContent) return;
          el.classList.add('clickable');
          el.addEventListener('click', () => {
            const ply = Number(el.dataset.ply);
            this.reviewGo(ply);   // loads the game first if needed
          });
        });
      }
      this.historyList.appendChild(row);
    }

    // Keep the newest move, or the move being reviewed, in view
    const focus = this.historyList.querySelector('.reviewing');
    if (focus) focus.scrollIntoView({ block: 'nearest' });
    else this.historyList.scrollTop = this.historyList.scrollHeight;
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
    const plies = this.gameState.ply_count || 0;
    const moves = Math.ceil(plies / 2);
    this.gameOverDesc.textContent =
      `${this.formatTerminalReason(reason)} after ${moves} move${moves === 1 ? '' : 's'} (${plies} plies).`;
    const chat = this.currentGame === 'chaturanga' ? '' : 'none';
    document.getElementById('btn-modal-review').style.display = chat;
    document.getElementById('btn-modal-analyse').style.display = chat;
    // Once closed for review or analysis, don't cover the board again this game
    this.gameOverOverlay.style.display =
      this.reviewIndex === null && !this.overlayDismissed ? 'flex' : 'none';
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
