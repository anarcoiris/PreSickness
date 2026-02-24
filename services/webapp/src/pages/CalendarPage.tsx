import { useState, useMemo } from 'react';
import { useQuery } from '@tanstack/react-query';
import {
    format,
    startOfMonth,
    endOfMonth,
    startOfWeek,
    endOfWeek,
    addDays,
    addMonths,
    subMonths,
    isSameMonth,
    isSameDay,
    isToday,
} from 'date-fns';
import { es } from 'date-fns/locale';
import {
    ChevronLeft,
    ChevronRight,
    Heart,
    AlertCircle,
    Pill,
} from 'lucide-react';

import { eventsApi } from '../api/client';
import type { Event } from '../api/client';

const EVENT_TYPE_COLORS: Record<string, string> = {
    symptom_onset: 'var(--color-symptom)',
    confirmed_relapse: 'var(--color-relapse)',
    medication_start: 'var(--color-medication)',
    hospital_visit: 'var(--color-hospital)',
    doctor_appointment: 'var(--color-appointment)',
};

export default function CalendarPage() {
    const [currentMonth, setCurrentMonth] = useState(new Date());
    const [selectedDate, setSelectedDate] = useState<Date | null>(null);

    const { data: events = [] } = useQuery({
        queryKey: ['events'],
        queryFn: () => eventsApi.list().then(r => r.data),
    });

    // Crear mapa de eventos por fecha
    const eventsByDate = useMemo(() => {
        const map = new Map<string, Event[]>();
        events.forEach(event => {
            const dateKey = format(new Date(event.event_date), 'yyyy-MM-dd');
            if (!map.has(dateKey)) {
                map.set(dateKey, []);
            }
            map.get(dateKey)!.push(event);
        });
        return map;
    }, [events]);

    // Generar días del calendario
    const calendarDays = useMemo(() => {
        const monthStart = startOfMonth(currentMonth);
        const monthEnd = endOfMonth(monthStart);
        const startDate = startOfWeek(monthStart, { weekStartsOn: 1 });
        const endDate = endOfWeek(monthEnd, { weekStartsOn: 1 });

        const days: Date[] = [];
        let day = startDate;
        while (day <= endDate) {
            days.push(day);
            day = addDays(day, 1);
        }
        return days;
    }, [currentMonth]);

    const selectedEvents = selectedDate
        ? eventsByDate.get(format(selectedDate, 'yyyy-MM-dd')) || []
        : [];

    return (
        <div>
            <div className="page-header">
                <div>
                    <h1 className="page-title">Calendario de Eventos</h1>
                    <p className="page-subtitle">Visualiza tus eventos clínicos en el tiempo</p>
                </div>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: '2fr 1fr', gap: 'var(--space-xl)' }}>
                {/* Calendar */}
                <div className="card">
                    <div className="calendar-header">
                        <button
                            className="btn btn-icon btn-secondary"
                            onClick={() => setCurrentMonth(subMonths(currentMonth, 1))}
                        >
                            <ChevronLeft size={20} />
                        </button>
                        <h2 style={{ textTransform: 'capitalize' }}>
                            {format(currentMonth, 'MMMM yyyy', { locale: es })}
                        </h2>
                        <button
                            className="btn btn-icon btn-secondary"
                            onClick={() => setCurrentMonth(addMonths(currentMonth, 1))}
                        >
                            <ChevronRight size={20} />
                        </button>
                    </div>

                    <div className="calendar-grid">
                        {['Lun', 'Mar', 'Mié', 'Jue', 'Vie', 'Sáb', 'Dom'].map(day => (
                            <div key={day} className="calendar-day-header">
                                {day}
                            </div>
                        ))}

                        {calendarDays.map((day, idx) => {
                            const dateKey = format(day, 'yyyy-MM-dd');
                            const dayEvents = eventsByDate.get(dateKey) || [];
                            const isCurrentMonth = isSameMonth(day, currentMonth);
                            const isSelected = selectedDate && isSameDay(day, selectedDate);

                            // Determinar tipo de evento principal para el color
                            const hasRelapse = dayEvents.some(e => e.event_type === 'confirmed_relapse');
                            const hasSymptom = dayEvents.some(e => e.event_type === 'symptom_onset');
                            const hasMedication = dayEvents.some(e => e.event_type === 'medication_start');

                            return (
                                <div
                                    key={idx}
                                    className={`calendar-day ${isToday(day) ? 'today' : ''} ${dayEvents.length > 0 ? 'has-event' : ''}`}
                                    style={{
                                        opacity: isCurrentMonth ? 1 : 0.3,
                                        background: isSelected ? 'rgba(49, 130, 206, 0.3)' : undefined,
                                        cursor: 'pointer',
                                    }}
                                    onClick={() => setSelectedDate(day)}
                                >
                                    <span>{format(day, 'd')}</span>

                                    {/* Event indicators */}
                                    {dayEvents.length > 0 && (
                                        <div style={{
                                            display: 'flex',
                                            gap: '2px',
                                            position: 'absolute',
                                            bottom: '4px'
                                        }}>
                                            {hasRelapse && (
                                                <div style={{
                                                    width: '6px',
                                                    height: '6px',
                                                    borderRadius: '50%',
                                                    background: 'var(--color-relapse)',
                                                }} />
                                            )}
                                            {hasSymptom && (
                                                <div style={{
                                                    width: '6px',
                                                    height: '6px',
                                                    borderRadius: '50%',
                                                    background: 'var(--color-symptom)',
                                                }} />
                                            )}
                                            {hasMedication && (
                                                <div style={{
                                                    width: '6px',
                                                    height: '6px',
                                                    borderRadius: '50%',
                                                    background: 'var(--color-medication)',
                                                }} />
                                            )}
                                        </div>
                                    )}
                                </div>
                            );
                        })}
                    </div>

                    {/* Legend */}
                    <div style={{
                        display: 'flex',
                        gap: 'var(--space-lg)',
                        marginTop: 'var(--space-lg)',
                        paddingTop: 'var(--space-md)',
                        borderTop: '1px solid rgba(255,255,255,0.1)',
                    }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-sm)' }}>
                            <div style={{ width: 12, height: 12, borderRadius: '50%', background: 'var(--color-relapse)' }} />
                            <span style={{ fontSize: '0.875rem', color: 'var(--color-text-secondary)' }}>Brote</span>
                        </div>
                        <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-sm)' }}>
                            <div style={{ width: 12, height: 12, borderRadius: '50%', background: 'var(--color-symptom)' }} />
                            <span style={{ fontSize: '0.875rem', color: 'var(--color-text-secondary)' }}>Síntomas</span>
                        </div>
                        <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-sm)' }}>
                            <div style={{ width: 12, height: 12, borderRadius: '50%', background: 'var(--color-medication)' }} />
                            <span style={{ fontSize: '0.875rem', color: 'var(--color-text-secondary)' }}>Medicación</span>
                        </div>
                    </div>
                </div>

                {/* Event Details Panel */}
                <div className="card">
                    <div className="card-header">
                        <h2 className="card-title">
                            {selectedDate
                                ? format(selectedDate, "d 'de' MMMM", { locale: es })
                                : 'Selecciona una fecha'}
                        </h2>
                    </div>

                    {!selectedDate ? (
                        <div className="empty-state">
                            <p>Haz clic en un día para ver sus eventos</p>
                        </div>
                    ) : selectedEvents.length === 0 ? (
                        <div className="empty-state">
                            <p>No hay eventos en esta fecha</p>
                        </div>
                    ) : (
                        <div className="event-list">
                            {selectedEvents.map(event => (
                                <div key={event.id} style={{
                                    padding: 'var(--space-md)',
                                    background: 'rgba(255,255,255,0.05)',
                                    borderRadius: 'var(--radius-md)',
                                    borderLeft: `4px solid ${EVENT_TYPE_COLORS[event.event_type]}`,
                                }}>
                                    <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-sm)', marginBottom: 'var(--space-sm)' }}>
                                        {event.event_type === 'confirmed_relapse' && <Heart size={16} color="var(--color-relapse)" />}
                                        {event.event_type === 'symptom_onset' && <AlertCircle size={16} color="var(--color-symptom)" />}
                                        {event.event_type === 'medication_start' && <Pill size={16} color="var(--color-medication)" />}
                                        <strong>
                                            {event.event_type === 'confirmed_relapse' ? 'Brote Confirmado' :
                                                event.event_type === 'symptom_onset' ? 'Inicio de Síntomas' :
                                                    event.event_type === 'medication_start' ? 'Inicio Medicación' :
                                                        event.event_type}
                                        </strong>
                                    </div>

                                    {event.severity && (
                                        <span className={`badge ${event.severity}`} style={{ marginBottom: 'var(--space-sm)' }}>
                                            {event.severity === 'mild' ? 'Leve' :
                                                event.severity === 'moderate' ? 'Moderado' : 'Severo'}
                                        </span>
                                    )}

                                    {event.notes && (
                                        <p style={{
                                            margin: 0,
                                            fontSize: '0.875rem',
                                            color: 'var(--color-text-secondary)',
                                            marginTop: 'var(--space-sm)',
                                        }}>
                                            {event.notes}
                                        </p>
                                    )}
                                </div>
                            ))}
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
}
