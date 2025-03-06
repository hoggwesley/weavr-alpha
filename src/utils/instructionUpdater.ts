import * as readline from 'readline';

export async function updateInstructions(): Promise<string> {
    const rl = readline.createInterface({
        input: process.stdin,
        output: process.stdout,
    });

    return new Promise((resolve) => {
        let newInstructions = '';
        console.log("Enter new instructions (press Ctrl+D on a new line when finished):");

        rl.on('line', (line) => {
            newInstructions += line + '\n';
        });

        rl.on('close', () => {
            console.log("Readline interface closed.");
            resolve(newInstructions.trim());
        });

        rl.on('SIGINT', () => {
            console.log("SIGINT received, closing readline interface.");
            rl.close();
        });
    });
}
