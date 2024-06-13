"use client";
import { useState } from "react";
import Select from "react-select";

export default function Home() {
  const [categories, setCategories] = useState([]);
  const [selectedCity, setSelectedCity] = useState(null);
  const [recommendedEventTitles, setRecommendedEventTitles] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const categoryOptions = [
    { value: "Musik", label: "Musik" },
    { value: "Education", label: "Education" },
    { value: "Olahraga", label: "Olahraga" },
    { value: "Budaya", label: "Budaya" },
    { value: "Wisata", label: "Wisata" },
    { value: "Bisnis", label: "Bisnis" },
    { value: "Entertainment", label: "Entertainment" },
    { value: "Lainnya", label: "Lainnya" },
  ];

  const cityOptions = [
    { value: "Bali", label: "Bali" },
    { value: "DKI Jakarta", label: "DKI Jakarta" },
    { value: "Bandung", label: "Bandung" },
    { value: "Semarang", label: "Semarang" },
    { value: "Surakarta", label: "Surakarta" },
    { value: "DI Yogyakarta", label: "DI Yogyakarta" },
  ];

  const fetchRecommendations = async () => {
    setLoading(true);
    try {
      const response = await fetch("http://localhost:5000/recommend", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          Category: categories.map((category) => category.value),
          Location_City: selectedCity?.value,
        }),
      });
      const data = await response.json();
      if (data.error) {
        setError(data.error);
        setRecommendedEventTitles([]);
      } else {
        setRecommendedEventTitles(data.recommended_events);
        setError("");
      }
    } catch (error) {
      setError("An error occurred. Please try again later.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex flex-col items-center justify-center min-h-screen py-2">
      <h1 className="text-3xl font-bold mb-4">Event Recommendation</h1>
      <div className="w-full max-w-lg">
        <label htmlFor="categories" className="block text-gray-700 mb-2">
          Select event categories:
        </label>
        <Select
          id="categories"
          className="mb-4"
          options={categoryOptions}
          isMulti
          value={categories}
          onChange={setCategories}
        />
        <label htmlFor="city" className="block text-gray-700 mb-2">
          Select a city:
        </label>
        <Select
          id="city"
          className="mb-4"
          options={cityOptions}
          value={selectedCity}
          onChange={setSelectedCity}
          isClearable
        />
        <button
          className="bg-blue-500 text-white px-4 py-2 rounded"
          onClick={fetchRecommendations}
          disabled={loading}
        >
          {loading ? "Fetching..." : "Fetch Recommendations"}
        </button>
        {error && <p className="text-red-500 mt-2">{error}</p>}
        {recommendedEventTitles.length > 0 && (
          <div className="mt-4">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {recommendedEventTitles.map((event, index) => (
                <div key={index} className="bg-white shadow-md rounded-lg p-4">
                  <p className="font-bold text-md mb-5">{event.Title}</p>
                  <p className="font-bold">{event.Location_City}</p>
                  <p className="text-gray-600">{event.Category}</p>
                  <p className="text-gray-600">{event.Event_Start}</p>
                  <p className="text-gray-800">{event.Location_Address}</p>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
