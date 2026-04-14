# 🌾 AgriSaarthi

> **Your AI-powered digital companion for Indian agriculture.**

AgriSaarthi ("Saarthi" meaning *guide* in Hindi) is a smart, accessible web application built to empower Indian farmers with real-time agricultural insights, AI-driven assistance, and a localized experience — all in one platform.

---

## ✨ Features

### 🌐 Multi-Language Support
Full localization in **English**, **Hindi**, and **Gujarati** — ensuring every farmer can use the platform in their preferred language.

### 👤 Role-Based Access
Tailored dashboards and functionalities for four distinct user roles:
- 🧑‍🌾 **Farmers** — Core user experience with crop, weather, and market tools
- 🏛️ **Agriculture Officers** — Oversight and advisory tools
- 🛒 **Vendors** — Market and supply-chain features
- ⚙️ **Administrators** — Platform management

### 🔐 Intuitive Authentication
- Traditional **phone/password login**
- Innovative **voice-based login** for accessibility

### 🤖 AI-Powered Voice Assistant
Integrated with **Vapi.ai** to enable natural-language, voice-driven queries about farming — no typing required.

### 📊 Personalized Dashboard
Quick-access cards for the most critical farming needs:
| Feature | Description |
|---|---|
| 🌿 Crop Health | Disease and pest detection via camera |
| 🌦️ Weather Forecasts | Current, hourly, and weekly forecasts with alerts |
| 📈 Live Market Prices | Real-time commodity pricing |
| 🏛️ Government Schemes | Latest subsidies and programs |
| 💧 Irrigation | Smart irrigation management |

### 📷 Quick Crop Analysis
Use the device camera to perform **instant disease and pest detection** on crops.

### 🔔 Smart Notifications
Timely alerts for weather changes, market price fluctuations, and new government scheme announcements.

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **Frontend** | React, TypeScript |
| **Styling** | Tailwind CSS |
| **Animations** | Framer Motion |
| **Routing** | React Router DOM |
| **Icons** | Lucide React |
| **State Management** | React Context API |
| **Backend / Auth / DB** | Firebase (Auth + Realtime Database) |
| **AI Voice Assistant** | Vapi.ai Web SDK |

---

## 🚀 Getting Started

### Prerequisites
- Node.js (v18+)
- npm or yarn
- Firebase project with Auth and Realtime Database enabled
- Vapi.ai account for voice assistant integration

### Installation

```bash
# Clone the repository
git clone https://github.com/1HPdhruv/AgriSaarthi.git
cd AgriSaarthi

# Install dependencies
npm install
```

### Environment Setup

Create a `.env` file in the root directory and populate it with your keys:

```env
VITE_FIREBASE_API_KEY=your_firebase_api_key
VITE_FIREBASE_AUTH_DOMAIN=your_project.firebaseapp.com
VITE_FIREBASE_DATABASE_URL=https://your_project-default-rtdb.firebaseio.com
VITE_FIREBASE_PROJECT_ID=your_project_id
VITE_FIREBASE_STORAGE_BUCKET=your_project.appspot.com
VITE_FIREBASE_MESSAGING_SENDER_ID=your_sender_id
VITE_FIREBASE_APP_ID=your_app_id
VITE_VAPI_PUBLIC_KEY=your_vapi_public_key
```

### Running Locally

```bash
npm run dev
```

The app will be available at `http://localhost:5173`.

### Building for Production

```bash
npm run build
```

---

## 🗂️ Project Structure

```
AgriSaarthi/
├── public/
├── src/
│   ├── components/       # Reusable UI components
│   ├── context/          # React Context providers (Auth, Language)
│   ├── pages/            # Route-level page components
│   ├── hooks/            # Custom React hooks
│   ├── utils/            # Helper functions
│   └── main.tsx          # Application entry point
├── .env                  # Environment variables (not committed)
├── tailwind.config.ts
├── tsconfig.json
└── vite.config.ts
```

---

## 🌍 Localization

AgriSaarthi supports three languages out of the box. Language preference is persisted via React Context and stored per user profile.

| Language | Code |
|---|---|
| English | `en` |
| Hindi | `hi` |
| Gujarati | `gu` |

---

## 🔭 Roadmap

- [ ] Expanded AI assistant with deeper agricultural knowledge base
- [ ] Image recognition for advanced crop disease diagnosis
- [ ] Live API integration for real-time market prices and weather
- [ ] Personalized irrigation schedules based on soil moisture
- [ ] Direct farmer-to-buyer marketplace
- [ ] Android & iOS native apps
- [ ] Community forums and expert Q&A
- [ ] Farm performance analytics and yield predictions

---

## 🏆 Recognition

AgriSaarthi was built and submitted to the [**World's Largest Hackathon presented by Bolt**](https://worldslargesthackathon.devpost.com/) on Devpost.

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create your feature branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m 'Add your feature'`
4. Push to the branch: `git push origin feature/your-feature`
5. Open a Pull Request

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

## 👨‍💻 Team

Built with ❤️ for Indian farmers.

| Name | GitHub |
|---|---|
| Dhruv | [@1HPdhruv](https://github.com/1HPdhruv) |

---

<p align="center">
  <i>"Every farmer deserves a Saarthi."</i>
</p>
