document.addEventListener('DOMContentLoaded', () => {
    const loginScreen = document.getElementById('login-screen');
    const signupScreen = document.getElementById('signup-screen');
    const transactionsScreen = document.getElementById('transactions-screen');
    const loginBtn = document.getElementById('login-btn');
    const signupBtn = document.getElementById('signup-btn');
    const logoutBtn = document.getElementById('logout-btn');
    const showSignUpBtn = document.getElementById('show-signup-btn');
    const showLoginBtn = document.getElementById('show-login-btn');
    const loginEmail = document.getElementById('email-login');
    const loginPassword = document.getElementById('password-login');
    const signupEmail = document.getElementById('email-signup');
    const signupPassword = document.getElementById('password-signup');

    let users = [];  // User storage

    const transactions = [
        { item: 'Boba', transactionType: 'Check In', date: '24 Nov 2024', location: 'A1' },
        { item: 'Snow Ice', transactionType: 'Check In', date: '24 Nov 2024', location: 'B2' },
        { item: 'Boba', transactionType: 'Check In', date: '25 Nov 2024', location: 'A1' }
    ];

    function populateTransactions() {
        const tableBody = document.getElementById('transactions-table');
        tableBody.innerHTML = '';
        transactions.forEach(t => {
            const row = document.createElement('tr');
            row.innerHTML = `<td>${t.item}</td><td>${t.transactionType}</td><td>${t.date}</td><td>${t.location}</td>`;
            tableBody.appendChild(row);
        });
    }

    function showScreen(screen) {
        [loginScreen, signupScreen, transactionsScreen].forEach(s => {
            s.style.display = 'none';
        });
        screen.style.display = 'block';
    }

    loginBtn.addEventListener('click', () => {
        const user = users.find(u => u.email === loginEmail.value && u.password === loginPassword.value);
        if (user) {
            showScreen(transactionsScreen);
            populateTransactions();
        } else {
            alert('Invalid login credentials. Please try again.');
        }
    });

    signupBtn.addEventListener('click', () => {
        if (users.some(u => u.email === signupEmail.value)) {
            alert('An account with this email already exists.');
        } else {
            users.push({ email: signupEmail.value, password: signupPassword.value });
            alert('Account created successfully!');
            showScreen(loginScreen);
        }
    });

    logoutBtn.addEventListener('click', () => showScreen(loginScreen));
    showSignUpBtn.addEventListener('click', () => showScreen(signupScreen));
    showLoginBtn.addEventListener('click', () => showScreen(loginScreen));
});
