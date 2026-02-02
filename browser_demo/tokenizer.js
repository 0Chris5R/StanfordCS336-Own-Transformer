/**
 * BPE Tokenizer for browser - JavaScript port of the Python tokenizer
 */

class BPETokenizer {
    constructor() {
        this.vocab = {};          // id -> bytes (as Uint8Array)
        this.reverseVocab = {};   // bytes (as string key) -> id
        this.merges = [];         // [[id1, id2], ...]
        this.specialTokens = [];
        this.mergeLookup = {};    // "id1,id2" -> [rank, mergedId]
    }

    async load(tokenizerPath) {
        const response = await fetch(tokenizerPath);
        const data = await response.json();

        // Build vocab
        for (const [idStr, byteList] of Object.entries(data.vocab)) {
            const id = parseInt(idStr);
            this.vocab[id] = new Uint8Array(byteList);
        }

        // Build reverse vocab
        for (const [id, bytes] of Object.entries(this.vocab)) {
            this.reverseVocab[this._bytesToKey(bytes)] = parseInt(id);
        }

        // Load merges
        this.merges = data.merges;

        // Build merge lookup for efficient encoding
        for (let rank = 0; rank < this.merges.length; rank++) {
            const [a, b] = this.merges[rank];
            const mergedBytes = this._concatBytes(this.vocab[a], this.vocab[b]);
            const mergedId = this.reverseVocab[this._bytesToKey(mergedBytes)];
            this.mergeLookup[`${a},${b}`] = [rank, mergedId];
        }

        // Load special tokens
        this.specialTokens = data.special_tokens || [];

        console.log(`Tokenizer loaded: vocab=${Object.keys(this.vocab).length}, merges=${this.merges.length}`);
    }

    _bytesToKey(bytes) {
        return Array.from(bytes).join(',');
    }

    _concatBytes(a, b) {
        const result = new Uint8Array(a.length + b.length);
        result.set(a, 0);
        result.set(b, a.length);
        return result;
    }

    encode(text) {
        const encoder = new TextEncoder();
        const ids = [];

        // Handle special tokens by splitting on them
        let parts = [text];
        if (this.specialTokens.length > 0) {
            const sortedTokens = [...this.specialTokens].sort((a, b) => b.length - a.length);
            const pattern = new RegExp(`(${sortedTokens.map(t => this._escapeRegex(t)).join('|')})`, 'g');
            parts = text.split(pattern).filter(p => p !== '');
        }

        for (const part of parts) {
            if (this.specialTokens.includes(part)) {
                // Encode special token as single ID
                const tokenBytes = encoder.encode(part);
                const key = this._bytesToKey(tokenBytes);
                if (this.reverseVocab[key] !== undefined) {
                    ids.push(this.reverseVocab[key]);
                }
                continue;
            }

            // Pre-tokenize with regex (simplified version)
            const chunks = this._pretokenize(part);

            for (const chunk of chunks) {
                const bytes = encoder.encode(chunk);
                // Convert each byte to its token ID
                let chunkIds = Array.from(bytes).map(b => b);

                // Apply BPE merges
                while (chunkIds.length >= 2) {
                    let bestPair = null;
                    let bestRank = Infinity;

                    for (let i = 0; i < chunkIds.length - 1; i++) {
                        const key = `${chunkIds[i]},${chunkIds[i + 1]}`;
                        if (this.mergeLookup[key]) {
                            const [rank, _] = this.mergeLookup[key];
                            if (rank < bestRank) {
                                bestRank = rank;
                                bestPair = [i, chunkIds[i], chunkIds[i + 1]];
                            }
                        }
                    }

                    if (bestPair === null) break;

                    const [pos, a, b] = bestPair;
                    const [_, mergedId] = this.mergeLookup[`${a},${b}`];
                    chunkIds = [
                        ...chunkIds.slice(0, pos),
                        mergedId,
                        ...chunkIds.slice(pos + 2)
                    ];
                }

                ids.push(...chunkIds);
            }
        }

        return ids;
    }

    _pretokenize(text) {
        // Simplified regex pattern matching the Python version
        const pattern = /'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+/gu;
        const matches = text.match(pattern);
        return matches || [];
    }

    _escapeRegex(string) {
        return string.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    }

    decode(ids) {
        const bytes = [];
        for (const id of ids) {
            if (this.vocab[id]) {
                bytes.push(...this.vocab[id]);
            }
        }
        const decoder = new TextDecoder('utf-8', { fatal: false });
        return decoder.decode(new Uint8Array(bytes));
    }

    getVocabSize() {
        return Object.keys(this.vocab).length;
    }
}

// Export for use in app.js
if (typeof window !== 'undefined') {
    window.BPETokenizer = BPETokenizer;
}
